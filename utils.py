import torch
import pandas as pd
from rdkit import Chem
from torch.utils.data import TensorDataset, random_split


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def process(all_sms):
    all_smiles = []
    element_table = ["Cl", "Br"]
    for i in range(len(all_sms)):
        sms = all_sms[i]
        smiles = []
        j = 0
        while j < len(sms):
            sms1 = []
            if sms[j] == "[":
                sms1.append(sms[j])
                j = j + 1
                while sms[j] != "]":
                    sms1.append(sms[j])
                    j = j + 1
                sms1.append(sms[j])
                sms2 = ''.join(sms1)
                smiles.append(sms2)
                j = j + 1
            else:
                sms1.append(sms[j])

                if j + 1 < len(sms):
                    sms1.append(sms[j + 1])
                    sms2 = ''.join(sms1)
                else:
                    sms1.insert(0, sms[j - 1])
                    sms2 = ''.join(sms1)

                if sms2 not in element_table:
                    smiles.append(sms[j])
                    j = j + 1
                else:
                    smiles.append(sms2)
                    j = j + 2

        all_smiles.append(list(smiles))
    return all_smiles


class Corpus(object):
    def __init__(self, sm_list):
        self.dictionary = Dictionary()
        self.all = self.tokenize(sm_list)

    def tokenize(self, sm_list):
        self.dictionary.add_word('\n')
        all_smiles = process(sm_list)

        all_ids = []
        for smiles in all_smiles:
            id = []
            words = ['&'] + smiles
            for word in words:
                self.dictionary.add_word(word)
                id += [self.dictionary.word2idx[word]]

            while len(id) < 142:
                id += [0]

            all_ids.append(id)
        print(self.dictionary.word2idx)
        return all_ids


if __name__ == "__main__":
    data = pd.read_csv('data/opv.csv')
    corpus = Corpus(data.loc[:]['smiles'])
    print(corpus.dictionary)
    torch.save([corpus.dictionary.word2idx, corpus.dictionary.idx2word], "data/opv_dic.pt")

    sms = []
    inputs = []
    targets = []
    count = np.random.permutation(len(data.index))
    for i in count:
        smiles = data.loc[i]['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if data.loc[i]['PCE'] > 0.5:
            sms.append(smiles + '\n')
            inputs.append(corpus.all[i])
            targets.append([data.loc[i]['homo'], data.loc[i]['lumo'],
                            data.loc[i]['homo_calib'], data.loc[i]['lumo_calib'],
                            data.loc[i]['PCE'], data.loc[i]['PCE_calib']])

    with open("data/smi_c.txt", "w") as f:
        f.writelines(sms)

    all_data = TensorDataset(torch.LongTensor(inputs), torch.FloatTensor(targets))
    train_data, vali_data, test_data = all_data[:-4000],all_data[-4000:-2000],all_data[-2000:]
    torch.save([train_data, vali_data, test_data], "results/saved_models/opv_data.pt")
    print(inputs[-1], targets[-1])
