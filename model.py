import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, encoder_dim=512, enc_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = enc_image_size

        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2] # remove fc and pool layers
        self.resnet = nn.Sequential(*modules)
        self.conv1x1 = nn.Conv2d(in_channels=2048, out_channels=encoder_dim, kernel_size=1) # reduce dims to 512
        self.adaptive_pool = nn.AdaptiveAvgPool2d((enc_image_size, enc_image_size))
        
        # fine-tune params in last convolutional block
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in list(self.resnet.children())[-1].parameters():
            param.requires_grad = True

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)
        out = self.adaptive_pool(out)
        out = self.conv1x1(out)
        out = out.permute(0, 2, 3, 1)  # (batch_size, enc_image_size, enc_image_size, 512)
        return out


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_attn = nn.Linear(encoder_dim, attention_dim)
        self.decoder_attn = nn.Linear(decoder_dim, attention_dim)
        self.full_attn = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_out, decoder_hidden):
        encoder_attn_weights = self.encoder_attn(encoder_out)
        decoder_attn_weights = self.decoder_attn(decoder_hidden)
        attn = self.relu(encoder_attn_weights + decoder_attn_weights.unsqueeze(1))
        attn = self.full_attn(attn).squeeze(2)
        probs = F.softmax(attn, dim=1)
        enc_with_attn = (encoder_out * probs.unsqueeze(2)).sum(dim=1)
        return enc_with_attn, probs


class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=512, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.dropout = nn.Dropout(p=self.dropout)
        self.lstm_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, cap_len):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param cap_len: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort by decreasing length
        cap_len = torch.LongTensor(cap_len)
        cap_len, sort_ind = cap_len.sort(dim=0, descending=True)
        dec_len = (cap_len - 1).tolist()

        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(dec_len), self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(dec_len), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(dec_len)):
            batch_size_t = sum([l > t for l in dec_len])
            enc_with_attn, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = F.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            enc_with_attn = gate * enc_with_attn
            h, c = self.lstm_step(
                    torch.cat([embeddings[:batch_size_t, t, :], enc_with_attn], dim=1),
                    (h[:batch_size_t], c[:batch_size_t])
                )  # (batch_size_t, decoder_dim)
            h = self.dropout(h)
            preds = self.fc(h)  # (batch_size_t, vocab_size)
            
            alphas[:batch_size_t, t, :] = alpha
            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, dec_len, alphas, sort_ind
