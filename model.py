import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, encoder_dim, enc_image_size=14):
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
    def __init__(self, attention_dim, embed_dim, decoder_dim, encoder_dim, vocab_size, dropout=0.5):
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
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, encoder_out):
        enc_mean = encoder_out.mean(dim=1)
        h = self.init_h(enc_mean)
        c = self.init_c(enc_mean)
        return h, c

    def forward(self, encoder_out, enc_cap, cap_len):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        h, c = self.init_hidden(encoder_out)

        # Sort by decreasing length
        cap_len = torch.LongTensor(cap_len)
        cap_len, sort_ind = cap_len.sort(dim=0, descending=True)
        dec_len = (cap_len - 1).tolist()

        encoder_sorted = encoder_out[sort_ind]
        caps_sorted = enc_cap[sort_ind]
        embeddings = self.embedding(caps_sorted)

        output = torch.zeros(batch_size, max(dec_len), self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(dec_len), num_pixels).to(device)

        for t in range(max(dec_len)):
            bs = sum([l > t for l in dec_len])
            enc_with_attn, alpha = self.attention(encoder_sorted[:bs], h[:bs])
            gate = F.sigmoid(self.f_beta(h[:bs]))
            enc_with_attn = gate * enc_with_attn
            h, c = self.lstm_step(
                    torch.cat([embeddings[:bs, t, :], enc_with_attn], dim=1),
                    (h[:bs], c[:bs])
                )
            h = self.dropout(h)
            out = self.fc(h)
            
            alphas[:bs, t, :] = alpha
            output[:bs, t, :] = out

        return output, caps_sorted, dec_len, alphas, sort_ind
