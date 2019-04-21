import torch
from torchvision import transforms 
import pickle
from PIL import Image
import torch.nn.functional as F

class Sampler(object):
    def __init__(self, state, vocab, transform = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.load(state)
        self.vocab = vocab

        self.encoder = state['encoder'].to(self.device)
        self.decoder = state['decoder'].to(self.device)

        self.encoder.eval()
        self.decoder.eval()

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(), 
                transforms.Normalize((0.485, 0.456, 0.406), 
                                    (0.229, 0.224, 0.225))])

    def sample(self, img_path):
        image = Image.open(img_path)
        base_image = image

        if self.transform is not None:
            image = self.transform(image).unsqueeze(0)
        image = image.to(self.device)

        # Generate an caption from the image
        feature = self.encoder(image)
        sampled_ids = self.decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()

        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = self.vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)
        
        return base_image, sentence

    def beam_search(self, img_path, k=3):
        sequences = [[[], 1.0]]
        vocab_size = len(self.vocab)
        image = self.read_image(img_path)
        start_index = self.vocab.word2idx['<start>']
        end_index = self.vocab.word2idx['<end>']

        encoder_out = self.encoder(image)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)
        encoder_out = encoder_out.view(1, -1, encoder_dim)
        encoder_out = encoder_out.expand(k, encoder_out.size(1), encoder_dim)

        step_k_prev_words = torch.LongTensor([[start_index]] * k).to(self.device)
        sequences = step_k_prev_words

        top_k_scores = torch.zeros(k, 1).to(self.device)
        sequence_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(self.device)

        complete_sequence = []
        complete_sequence_alpha = []
        complete_sequence_scores = []

        # Begin decoding
        step = 1
        h, c = self.decoder.init_hidden(encoder_out)

        while True:
            embeddings = self.decoder.embedding(step_k_prev_words).squeeze(1)
            enc_with_attn, alpha = self.decoder.attention(encoder_out, h)
            alpha = alpha.view(-1, enc_image_size, enc_image_size)
            gate = F.sigmoid(self.decoder.f_beta(h))
            enc_with_attn = gate * enc_with_attn
            h, c = self.decoder.lstm_step(
                torch.cat([embeddings, enc_with_attn], dim=1),
                (h, c)
            )
            scores = self.decoder.fc(h)
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores

            if step == 1:
                top_k_scores, top_k_idx = scores[0].topk(k, dim=0)
            else:
                top_k_scores, top_k_idx = scores.view(-1).topk(k, dim=0)

            # Convert unrolled indices to actual indices of scores
            prev_word_idx = top_k_idx / vocab_size
            next_word_idx = top_k_idx % vocab_size

            # Add new words to sequences, alphas
            sequences = torch.cat([sequences[prev_word_idx], next_word_idx.unsqueeze(1)], dim=1)
            sequence_alpha = torch.cat([sequence_alpha[prev_word_idx], alpha[prev_word_idx].unsqueeze(1)], dim=1)

            incomplete_idx = [i for i, next_word in enumerate(next_word_idx) if next_word != end_index]
            complete_idx = list(set(range(len(next_word_idx))) - set(incomplete_idx))

            if len(complete_idx) > 0:
                complete_sequence.extend(sequences[complete_idx].tolist())
                complete_sequence_alpha.extend(sequence_alpha[complete_idx].tolist())
                complete_sequence_scores.extend(top_k_scores[complete_idx])
            k -= len(complete_idx)

            # Proceed with incomplete sequences
            if k == 0:
                break
            sequences = sequences[incomplete_idx]
            sequence_alpha = sequence_alpha[incomplete_idx]
            h = h[prev_word_idx[incomplete_idx]]
            c = c[prev_word_idx[incomplete_idx]]
            encoder_out = encoder_out[prev_word_idx[incomplete_idx]]
            top_k_scores = top_k_scores[incomplete_idx].unsqueeze(1)
            step_k_prev_words = next_word_idx[incomplete_idx].unsqueeze(1)

            if step > 50:
                break
            step += 1

        max_idx = complete_sequence_scores.index(max(complete_sequence_scores))
        seq = complete_sequence[max_idx]
        alphas = complete_sequence_alpha[max_idx]

        return seq, alphas


    def read_image(self, img_path):
        # Read image and process
        image = Image.open(img_path)
        base_image = image
        if self.transform is not None:
            image = self.transform(image).unsqueeze(0)
        image = image.to(self.device)
        return image
