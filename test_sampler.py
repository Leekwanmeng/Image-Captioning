from sampler import Sampler
from build_vocab import Vocabulary

model = Sampler('./models/checkpoint_1_5120.pt', './data/vocab.pkl')

# model.decoder.max_seg_length = 20

seq, alpha = model.caption_image_beam_search('./sample/sample_beach.jpg')

print(seq)