import torch
from torchvision import transforms 
import pickle
from PIL import Image
import torch.nn.functional as F

class Sampler(object):
    def __init__(self, state, vocab_path, transform = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state = torch.load(state)
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

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

    def caption_image_beam_search(self, img_path, beam_size=3):
        """
        Reads an image and captions it with beam search.
        :param encoder: encoder model
        :param decoder: decoder model
        :param image_path: path to image
        :param self.vocab: word map
        :param beam_size: number of sequences to consider at each decode-step
        :return: caption, weights for visualization
        """

        k = beam_size
        vocab_size = len(self.vocab)

        # Read image and process
        image = Image.open(img_path)
        base_image = image
        if self.transform is not None:
            image = self.transform(image).unsqueeze(0)
        image = image.to(self.device)

        # Encode
        encoder_out = self.encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[self.vocab.word2idx['<start>']]] * k).to(self.device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(self.device)  # (k, 1)

        # Tensor to store top k sequences' alphas; now they're just 1s
        seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(self.device)  # (k, 1, enc_image_size, enc_image_size)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        complete_seqs_alpha = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = self.decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = self.decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, alpha = self.decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

            gate = self.decoder.sigmoid(self.decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = self.decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences, alphas
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                                dim=1)  # (s, step+1, enc_image_size, enc_image_size)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                            next_word != self.vocab.word2idx['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]

        sentence = ""
        for i in seq[1:-1]:
            sentence += " " + self.vocab.idx2word[i]
        print(sentence)
        return seq, alphas