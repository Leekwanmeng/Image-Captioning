from sampler import Sampler
from build_vocab import Vocabulary


model = Sampler('./models/checkpoint_1_7142.pt', './data/vocab.pkl')

# model.decoder.max_seg_length = 20

seq, alpha = model.caption_image_beam_search('./sample/sample_dog.jpg')
seq, alpha = model.caption_image_beam_search('./sample/sample_beach.jpg')
seq, alpha = model.caption_image_beam_search('./sample/sample_img.jpg')
seq, alpha = model.caption_image_beam_search('./sample/sample_office.jpg')
# print(seq)

