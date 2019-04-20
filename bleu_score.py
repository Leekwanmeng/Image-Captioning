import torch
from torchvision import transforms 
import pickle
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
from data_loader import get_loader 
from build_vocab import Vocabulary
import os
import numpy as np
import os.path

def bleu_score(val_loader, encoder, decoder, device):
    encoder.eval()
    decoder.eval()

    ref = list()
    hyp = list()

    with torch.no_grad():
        for batch_idx, (img, target, lengths) in enumerate(val_loader):
            if batch_idx % 100 == 0:
                print("Progress: {:.0f} %".format(100. * batch_idx / len(val_loader)))

            img = img.to(device)
            target = target.to(device)
            
            encoder_out = encoder(img)
            
            output, caps_sorted, dec_len, alphas, sort_ind = decoder(encoder_out, target, lengths)
            target = caps_sorted[:, 1:]
            # target, _ = pack_padded_sequence(target, dec_len, batch_first=True)

            for i in target:
                i = i.tolist()
                ref.append([list(filter(lambda x: x != 0, i))])
        
            _, pred = torch.max(output, dim=2)
            pred = pred.tolist()
            temp_pred = list()
            for j, p in enumerate(pred):
                temp_pred.append(pred[j][:dec_len[j]])  # remove pads
            pred = temp_pred
            hyp.extend(pred)

            assert(len(hyp) == len(ref))
        bleu4 = corpus_bleu(ref, hyp)

    return bleu4

def main():
    checkpoint_path = './models/checkpoint_2_7340.pt'
    vocab_path = './data/vocab.pkl'

    caption_dir = 'data/annotations/'
    val_dir = 'data/val2014'
    batch_size = 32
    num_workers = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(checkpoint_path)
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    encoder = state['encoder'].to(device)
    decoder = state['decoder'].to(device)

    encoder.eval()
    decoder.eval()

    transform = transforms.Compose([ 
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    val_annotations = os.path.join(caption_dir , "captions_{}.json".format(os.path.basename(val_dir))) 
    val_loader = get_loader(val_dir, val_annotations, vocab, transform, batch_size, shuffle=True, num_workers=num_workers)

    # ids = val_loader.dataset.ids
    
    # val_loader.dataset.ids = np.random.choice(val_loader.dataset.ids, 2000, replace=False)

    print("Scoring model...")
    score = bleu_score(val_loader, encoder, decoder, device)

    print("BLEU-4 SCORE: {:.4f}".format(score*100))

if __name__ == "__main__":
    main()


