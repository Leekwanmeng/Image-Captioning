    
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from utils import AverageMeter

#TODO
def train(args, train_loader, device, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    encoder.train()
    decoder.train()

    loss_meter = AverageMeter('Train loss', ':.4f')
    top5acc = AverageMeter('Top 5 Accuracy', ':.4f')

    for batch_idx, (img, target) in enumerate(train_loader):
        img = img.to(device)
        target = target.to(device)
        # target = pack_padded_sequence(caption, length, batch_first=True)[0]

        encoder_out = encoder(img)
        output = decoder(encoder_out, target)
        loss = criterion(output, target)

        loss_meter.update(loss)

        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * args.batch_size, len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
        print('\nAverage train loss: {:.6f}'.format(loss_meter.avg))
    return

#TODO
def validate(args, val_loader, device, encoder, decoder, criterion):
    return

def main():
    parser = argparse.ArgumentParser(description='Image Caption Attn Model')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--encoder-lr', type=float, default=0.01, metavar='LR',
                        help='encoder learning rate (default: 0.01)')
    parser.add_argument('--decoder-lr', type=float, default=0.04, metavar='LR',
                        help='decoder learning rate (default: 0.04)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for dataloader')

    parser.add_argument('--hidden-dim', type=int, default=200, metavar='HD',
                        help='hidden dim (default: 200)')
    parser.add_argument('--lstm-layers', type=int, default=2, metavar='L',
                        help='num of lstm layers (default: 2)')
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--model_path', type=str, default='saved_models/' ,
                        help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--train_dir', type=str, default='data/train2014',
                        help='directory for train images')
    parser.add_argument('--val_dir', type=str, default='data/val2014',
                        help='directory for val images')
    parser.add_argument('--caption_dir', type=str, default='data/annotations/',
                        help='dir for annotation json file')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([ 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    #TODO
    vocab = ''

    train_annotations = os.path.join(args.caption_dir , "captions_{}.json".format(os.path.basename(args.train_dir))) 
    train_loader = get_loader(args.train_dir, train_annotations, vocab, transform, args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_annotations = os.path.join(args.caption_dir , "captions_{}.json".format(os.path.basename(args.val_dir))) 
    val_loader = get_loader(args.val_dir, val_annotations, vocab, transform, args.batch_size, shuffle=True, num_workers=args.num_workers)

    #TODO
    encoder = object.to(device)
    decoder = object.to(device)

    loss_fn = nn.CrossEntropyLoss()
    encoder_optim = torch.optim.Adam(params=encoder.parameters(), lr=args.encoder_lr)
    decoder_optim = torch.optim.Adam(params=decoder.parameters(), lr=args.decoder_lr)

    for epoch in range(1, (args.epochs + 1)):
        train_loss = train(args = args,
                train_loader=train_loader,
                device = device,
                encoder=encoder,
                decoder=decoder,
                criterion=loss_fn,
                encoder_optimizer=encoder_optim,
                decoder_optimizer=decoder_optim,
                epoch=epoch)

        val_loss, val_score = validate(args=args,
                val_loader=val_loader,
                device = device,
                encoder=encoder,
                decoder=decoder,
                criterion=loss_fn)

if __name__ == "__main__":
    main()