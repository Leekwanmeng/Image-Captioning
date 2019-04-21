from sampler import Sampler
from utils import Vocabulary


model = Sampler('./models/checkpoint_2_7369.pt', './data/vocab.pkl')

# model.decoder.max_seg_length = 20


s, seq, alpha = model.beam_search('./sample/sample_beach.jpg')
print(s)
s, seq, alpha = model.beam_search('./sample/sample_img.jpg')
print(s)
s, seq, alpha = model.beam_search('./sample/sample_office.jpg')
print(s)
s, seq, alpha = model.beam_search('./sample/sample_dog.jpg')
print(s)
# print(seq)

