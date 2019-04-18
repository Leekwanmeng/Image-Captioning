import torch
from torchvision import transforms 
import pickle
from PIL import Image

class Sampler(object):
    def __init__(self, state, vocab_path, transform = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        self.encoder = torch.load(state.encoder).to(self.device)
        self.decoder = torch.load(state.decoder).to(self.device)

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