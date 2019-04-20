import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import Encoder, DecoderWithAttention
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from utils import *
import time

def train(args, train_loader, device, encoder, decoder, criterion, optimizer, epoch):
    encoder.train()
    decoder.train()

    loss_meter = AverageMeter('Train loss', ':.4f')
    top5acc = AverageMeter('Top 5 Accuracy', ':.4f')
    epoch_start = time.time()
    last_time = epoch_start

    for batch_idx, (img, target, lengths) in enumerate(train_loader):
        img = img.to(device)
        target = target.to(device)

        encoder_out = encoder(img)
        
        output, caps_sorted, dec_len, alphas, sort_ind = decoder(encoder_out, target, lengths)
        target = caps_sorted[:, 1:]

        output, _ = pack_padded_sequence(output, dec_len, batch_first=True)
        target, _ = pack_padded_sequence(target, dec_len, batch_first=True)

        loss = criterion(output, target)

        # doubly stochastic attention regularization
        loss += 1. * ((1. - alphas.sum(dim=1)) ** 2).mean()
        loss_meter.update(loss)

        top5 = accuracy(output, target, 5)
        top5acc.update(top5, sum(dec_len))   

        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
    
        clip_gradient(optimizer, 5.)    

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            time_now = time.time()
            time_taken = time_now - last_time
            last_time = time_now
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Time taken: {:.0f}s\tLoss: {:.6f}\t Top 5 accuracy: {:.2f} %'.format(
            epoch, batch_idx * args.batch_size, len(train_loader.dataset),
            100. * batch_idx / len(train_loader), time_taken, loss.item(), top5acc.avg))
    print('\nAverage train loss: {:.6f}\t Top 5 accuracy: {:.2f} %\t Time taken: {:.0f}s'.format(loss_meter.avg, top5acc.avg, time.time() - epoch_start))
    return loss_meter.avg

def validate(args, val_loader, device, encoder, decoder, criterion):
    encoder.eval()
    decoder.eval()

    epoch_start = time.time()
    loss_meter = AverageMeter('Train loss', ':.4f')
    top5acc = AverageMeter('Top 5 Accuracy', ':.4f')
    print("Evaluating model...")
    epoch_start = time.time()
    with torch.no_grad():
        for batch_idx, (img, target, lengths) in enumerate(val_loader):
            img = img.to(device)
            target = target.to(device)
            
            encoder_out = encoder(img)
            
            output, caps_sorted, dec_len, alphas, sort_ind = decoder(encoder_out, target, lengths)
            target = caps_sorted[:, 1:]
            output, _ = pack_padded_sequence(output, dec_len, batch_first=True)
            target, _ = pack_padded_sequence(target, dec_len, batch_first=True)

            loss = criterion(output, target)
            # doubly stochastic attention regularization
            loss += 1. * ((1. - alphas.sum(dim=1)) ** 2).mean()
            loss_meter.update(loss)

            top5 = accuracy(output, target, 5)
            top5acc.update(top5, sum(dec_len))   
            if batch_idx % 100 == 0:
                print("Progress: {:.0f} %\t Time elapsed: {:.0f}s".format(100. * batch_idx / len(val_loader), time.time() - epoch_start))

    print(
        '\nAverage val loss: {:.6f}\t'
        'Time elapsed: {:.0f}s\t'
        'Top 5 accuracy: {:.4f} %\t'
        .format(loss_meter.avg, time.time() - epoch_start, top5acc.avg))
    return loss_meter.avg, top5acc.avg

def main():
    parser = argparse.ArgumentParser(description='Image Caption Attn Model')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--encoder-lr', type=float, default=0.001, metavar='LR',
                        help='encoder learning rate (default: 0.01)')
    parser.add_argument('--decoder-lr', type=float, default=0.001, metavar='LR',
                        help='decoder learning rate (default: 0.001)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of workers for dataloader')
    parser.add_argument('--crop-size', type=int, default=224,
                        help='size for randomly cropping images')

    parser.add_argument('--embed-dim', type=int, default=256, metavar='EMB',
                        help='embbed dim (default: 256)')
    parser.add_argument('--decoder-dim', type=int, default=512, metavar='HD',
                        help='decoder dim (default: 512)')
    parser.add_argument('--encoder-dim', type=int, default=512, metavar='HD',
                        help='encoder dim (default: 512)')
    parser.add_argument('--attention-dim', type=int, default=512, metavar='HD',
                        help='attention dim (default: 512)')
    parser.add_argument('--lstm-layers', type=int, default=1, metavar='L',
                        help='num of lstm layers (default: 1)')
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=30, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--model-path', type=str, default='saved_models/' ,
                        help='path for saving trained models')
    parser.add_argument('--vocab-path', type=str, default='data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--train-dir', type=str, default='data/train2014',
                        help='directory for train images')
    parser.add_argument('--val-dir', type=str, default='data/val2014',
                        help='directory for val images')
    parser.add_argument('--caption-dir', type=str, default='data/annotations/',
                        help='dir for annotation json file')
    parser.add_argument('--checkpoint-path', type=str, default=None)

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    #torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([ 
        transforms.RandomResizedCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    train_annotations = os.path.join(args.caption_dir , "captions_{}.json".format(os.path.basename(args.train_dir))) 
    train_loader = get_loader(args.train_dir, train_annotations, vocab, transform, args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_annotations = os.path.join(args.caption_dir , "captions_{}.json".format(os.path.basename(args.val_dir))) 
    val_loader = get_loader(args.val_dir, val_annotations, vocab, transform, args.batch_size, shuffle=True, num_workers=args.num_workers)

    encoder = Encoder(args.encoder_dim).to(device)
    decoder = DecoderWithAttention(args.attention_dim, args.embed_dim, args.decoder_dim, args.encoder_dim, len(vocab)).to(device)

    if args.checkpoint_path is not None:
        print('Loading from {}'.format(args.checkpoint_path))
        state = torch.load(args.checkpoint_path)
        encoder.load_state_dict(state['encoder'].state_dict())
        decoder.load_state_dict(state['decoder'].state_dict())


    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=[p for p in decoder.parameters() if p.requires_grad] + [p for p in encoder.parameters() if p.requires_grad], lr=args.decoder_lr)

    # train_loader.dataset.ids = train_loader.dataset.ids[:5000]
    # val_loader.dataset.ids = val_loader.dataset.ids[:20000]

    ids = val_loader.dataset.ids

    for epoch in range(1, (args.epochs + 1)):
        val_loader.dataset.ids = np.random.choice(val_loader.dataset.ids, 20000, replace=False)

        train_loss = train(args = args,
                train_loader=train_loader,
                device = device,
                encoder=encoder,
                decoder=decoder,
                criterion=loss_fn,
                optimizer=optimizer,
                epoch=epoch)

        val_loss, val_score = validate(args=args,
                val_loader=val_loader,
                device = device,
                encoder=encoder,
                decoder=decoder,
                criterion=loss_fn)

        filename = "./models/checkpoint_{}_{:.0f}.pt".format(epoch, val_score*100)
        state = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_score": val_score,
            "encoder": encoder,
            "decoder": decoder
        }
        
        torch.save(state, filename)
        print("saved model at {}\n".format(filename))

if __name__ == "__main__":
    main()
