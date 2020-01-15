import argparse
import time
import math
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import *
from rdkit import Chem


def evaluate(data_iter):
    model.eval()
    total_loss = 0

    for data, label in data_iter:
        targets = data[:, 1:]
        inputs = data[:, :-1]
        if torch.cuda.is_available()==True:
            targets=targets.cuda()
            inputs=inputs.cuda()

        outputs = model(inputs)

        final_output = outputs.contiguous().view(-1, n_words)
        final_target = targets.contiguous().view(-1)

        loss = criterion(final_output, final_target)
        total_loss += loss.item()

    return total_loss / len(data_iter)


def sample(idx2word, set_smi, num_samples):
    model.eval()
    n_words = len(idx2word)
    set_mols = [Chem.MolToInchiKey(Chem.MolFromSmiles(smi)) for smi in set_smi]
    n = 0
    new_mols = []
    new_smiles = []
    lss = 0
    for i in range(num_samples):
        input = torch.ones(1, 1, dtype=torch.long)
        if torch.cuda.is_available()==True:
            input=input.cuda()
        
        word = '&'
        while word[-1] != '\n':
            output = model(input)
            final_output = output.contiguous().view(-1, n_words)
            word_id = torch.multinomial(F.softmax(final_output[-1, :], dim=-1), num_samples=1).unsqueeze(0)
            input = torch.cat((input, word_id), dim=1)
            word += idx2word[word_id.item()]

        if bool(Chem.MolFromSmiles(word[1:])):
            n += 1
            mol = Chem.MolToInchiKey(Chem.MolFromSmiles(word[1:]))
            if mol not in set_mols and mol not in new_mols:
                new_mols += [mol]
                new_smiles += [word[1:]]
        if i != 0 and i % 10000 == 0:
            print(len(new_smiles) - lss)
            lss = len(new_smiles)
    print(n / num_samples)
    return new_smiles


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generative Modeling')
    parser.add_argument('--batch_size', type=int, default=32,
                        metavar='N', help='batch size (default: 32)')
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
    parser.add_argument('--levels', type=int, default=4,
                        help='# of levels (default: 4)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--nhid', type=int, default=256,
                        help='number of hidden units per layer (default: 256)')
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer type (default: Adam)')
    parser.add_argument('--save_name', type=str, default='gen.pt',
                        help='the name of save model')
    args = parser.parse_args()

    print(args)

    torch.manual_seed(1024)
    word2idx, idx2word = torch.load("data/opv_dic.pt")
    train_data, val_data, test_data = torch.load("data/opv_data.pt")
    train_iter = DataLoader(train_data, args.batch_size, shuffle=True)
    val_iter = DataLoader(val_data, args.batch_size, shuffle=False)
    test_iter = DataLoader(test_data, args.batch_size, shuffle=False)
    n_words = len(word2idx)

    model = GEN(args.emsize, n_words, hid_size=args.nhid, n_levels=args.levels,
                kernel_size=args.ksize, emb_dropout=args.emb_dropout, dropout=args.dropout )
    if torch.cuda.is_available()==True:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    best_vloss = 100
    try:
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            model.train()
            total_loss = 0

            for data, label in train_iter:
                targets = data[:, 1:]
                inputs = data[:, :-1]
                if torch.cuda.is_available()==True:
                    targets=targets.cuda()
                    inputs=inputs.cuda()

                optimizer.zero_grad()
                outputs = model(inputs)

                final_output = outputs.contiguous().view(-1, n_words)
                final_target = targets.contiguous().view(-1)

                loss = criterion(final_output, final_target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print('| epoch: {:3d} | train loss: {:5.6f} '.format
                    (epoch, total_loss / len(train_iter)))

            val_loss = evaluate(val_iter)
            scheduler.step(val_loss)

            print('-' * 89)
            print('| time: {:5.4f}s | valid loss: {:5.6f} | valid ppl: {:8.4f}'.format
                  ((time.time() - start_time), val_loss, math.exp(val_loss)))
            print('-' * 89)

            if val_loss < best_vloss:
                print('Save model!\n')
                torch.save(model.state_dict(), "results/saved_models/" + str(args.levels) + args.save_name)
                best_vloss = val_loss

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    model.load_state_dict(torch.load("results/saved_models/" + str(args.levels) + args.save_name), strict=True)
    test_loss = evaluate(test_iter)
    print('=' * 89)
    print('| End of training | test loss {:5.4f} | test ppl {:8.4f}'.format(test_loss, math.exp(test_loss)))
    print('=' * 89)

    with open('data/smi_c.txt', 'r') as smi:
        set_smiles = smi.readlines()
    new_smiles = sample(idx2word, set_smiles, num_samples=100000)
    with open("results/" + str(args.levels) + 'sample.txt', 'w') as f:
        f.writelines(new_smiles)
