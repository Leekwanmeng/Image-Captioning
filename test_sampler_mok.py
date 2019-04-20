from sampler_mok import Sampler
import os
import sys
import json

# For recreating a vocab class from scratch because there's an issue with 
# the pickle-saved vocab object
class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self, word2idx, idx2word):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

dirname = os.path.dirname(__file__)
with open(os.path.join(dirname, 'data', 'vocab_idx2word.json'), 'r') as f:
    vocab_idx2word = json.load(f)
with open(os.path.join(dirname, 'data', 'vocab_word2idx.json'), 'r') as f:
    vocab_word2idx = json.load(f)
vocab = Vocabulary(vocab_word2idx, vocab_idx2word)
model = Sampler(os.path.join(dirname, 'models', 'checkpoint_1_7142.pt'), vocab)

if __name__ == "__main__":
    # Test sampling
    sentence, _ = model.caption_image_beam_search('./sample/sample_img.jpg')
    print(sentence)

