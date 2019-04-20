from sampler import Sampler
from utils import Vocabulary


model = Sampler('./models/checkpoint_1_7149.pt', './data/vocab.pkl')

# model.decoder.max_seg_length = 20

seq, alpha = model.beam_search('./sample/sample_dog.jpg')
seq, alpha = model.beam_search('./sample/sample_beach.jpg')
seq, alpha = model.beam_search('./sample/sample_img.jpg')
seq, alpha = model.beam_search('./sample/sample_office.jpg')
# print(seq)

