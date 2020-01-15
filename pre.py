import argparse
import time
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from model import *


def evaluate(data_iter, args):
    model.eval()
    total_loss = 0

    for data, label in data_iter:
        targets = label[:, args.property_n:args.property_n + 1]
        inputs = data[:, 1:-1]
        if torch.cuda.is_available()==True:
            targets=targets.cuda()
            inputs=inputs.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
#        print(model.decoder.atten,outputs,targets)

    return total_loss / len(data_iter)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prediction Modeling')
    parser.add_argument('--property_n', type=int, default=0,
                        help='the numerical order of property (default: 0,1,2,3,4,5)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size (default: 32)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (default: 0.2)')
    parser.add_argument('--emb_dropout', type=float, default=0.1,
                        help='dropout applied to the embedded layer (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='upper epoch limit (default: 200)')
    parser.add_argument('--ksize', type=int, default=3,
                        help='kernel size (default: 3)')
    parser.add_argument('--emsize', type=int, default=32,
                        help='size of word embeddings (default: 32)')
    parser.add_argument('--levels', type=int, default=5,
                        help='# of levels (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--nhid', type=int, default=256,
                        help='number of hidden units per layer (default: 256)')
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer type (default: Adam)')
    parser.add_argument('--save_name', type=str, default='pre.pt',
                        help='the name of save model')
    args = parser.parse_args()

    print(args)

    torch.manual_seed(1024)
    word2idx, idx2word = torch.load("data/opv_dic.pt")
    train_data, val_data, test_data = torch.load("data/opv_data.pt")
    train_iter = DataLoader(train_data, args.batch_size, shuffle=True)
    val_iter = DataLoader(val_data, args.batch_size, shuffle=False)
    test_iter = DataLoader(test_data, 1, shuffle=False)

    n_words = len(word2idx)
    model = PRE(args.emsize, n_words, 1, hid_size=args.nhid, n_levels=args.levels,
                kernel_size=args.ksize, emb_dropout=args.emb_dropout,  dropout=args.dropout)
    
    if torch.cuda.is_available()==True:
        model.cuda()

    criterion = nn.MSELoss()
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    best_vloss = 100
    try:
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            model.train()
            total_loss = 0
            for data, label in train_iter:
                targets = label[:, args.property_n:args.property_n + 1]
                inputs = data[:, 1:-1]
                if torch.cuda.is_available()==True:
                    targets=targets.cuda()
                    inputs=inputs.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print('| epoch: {:3d} | train loss: {:5.6f} |'.format(epoch, total_loss / len(train_iter)))

            val_loss = evaluate(val_iter, args)
            scheduler.step(val_loss)

            print('-' * 89)
            print('| time: {:5.4f}s | valid loss: {:5.6f} |'.format((time.time() - start_time), val_loss))
            print('-' * 89)

            if val_loss < best_vloss:
                print('Save model!\n')
                torch.save(model, 'results/saved_models/' + str(args.levels) + str(args.property_n) + args.save_name)
                best_vloss = val_loss

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    model = torch.load('results/saved_models/' + str(args.levels) + str(args.property_n) + args.save_name)
    criterion = nn.L1Loss()
    val_L1_loss = evaluate(val_iter, args)
    test_L1_loss = evaluate(test_iter, args)
    print(val_L1_loss,test_L1_loss)
