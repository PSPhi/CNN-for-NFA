# -*- coding: UTF-8 -*-  
import torch
import numpy as np
import matplotlib
from utils import process
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
from pandas import DataFrame


def tok(ms, word2idx):
    all_ids = []
    all_smiles = process(ms)
    i = 0
    for smiles in all_smiles:
        i += 1
        id = []
        words = ['&'] + smiles

        if len(words) < 142:
            for word in words:
                id += [word2idx[word]]
            while len(id) < 142:
                id += [0]

            all_ids.append(id)
        else:
            print(i, words)

    return torch.LongTensor(all_ids)


def predict(ms, pt_path):
    word2idx, idx2word = torch.load("data/opv_dic.pt")
    data = tok(ms, word2idx)
    model = torch.load(pt_path, map_location=torch.device('cpu'))
    model.eval()
    out = []
    i = 0
    for input in data.unsqueeze(1):
        out += [model(input[:, 1:-1]).detach().item()]
        i += 1
    return out


def draw(x, y, z, x_range, y_range, label_x, label_y, label_z,save_name):
    plt.figure()
    plt.scatter(x, y, c=z, cmap='rainbow')
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.axis("square")
    plt.tick_params(direction='in', labelsize=16)
    plt.xlabel(label_x, {'weight': 'normal', 'size': 20})
    plt.ylabel(label_y, {'weight': 'normal', 'size': 20})
    plt.colorbar().set_label(label_z)
    figure_fig = plt.gcf()
    plt.tight_layout()
    figure_fig.savefig(save_name, format='eps', dpi=300)


def getfp(smiles):
    ms=[Chem.MolFromSmiles(smile) for smile in smiles]
    fps=[]
    for i in range(len(ms)):
        fps+=[AllChem.GetMorganFingerprint(ms[i],2)]
    return fps


if __name__ == '__main__':
    with open('results/5sample.txt', 'r') as f:
        ms = f.readlines()[:10000]
        fps = getfp(ms)

    h_pred = predict(ms, "results/saved_models/50pred.pt")
    l_pred = predict(ms, "results/saved_models/51pred.pt")
    p_pred = predict(ms, "results/saved_models/54pred.pt")

    draw(h_pred, l_pred, p_pred, x_range=(-7.4, -5.5), y_range=(-4.9, -1.8),
         label_x='$HOMO(eV)$', label_y='$LUMO(eV)$', label_z='PCE(%)', save_name='results/saved_models/figure7-1.eps')

    ori_df = pd.read_csv('data/opv.csv')
    with open('data/smi_c.txt', 'r') as f:
        ms0 = [m[:-1] for m in f.readlines()]
        fps0 = getfp(ms0)

    simi = []
    nn_list = []
    for i in range(len(ms)):
        simi_list = [DataStructs.DiceSimilarity(fps[i], fp0) for fp0 in fps0]
        max_simi = max(simi_list)
        simi += [max_simi * 100]
        nn_list += [ms0[simi_list.index(max_simi)]]

    nn_data = []
    for nn in nn_list:
        i = ori_df[ori_df['smiles'].isin([nn])]['index']
        nn_data.append([ori_df.loc[int(i), s] for s in ['homo','lumo','PCE']])

    draw(np.array(h_pred), np.array(nn_data)[:,0], np.array(simi),x_range=(-8, -5), y_range=(-8, -5),
         label_x='$HOMO_{gen}(eV)', label_y='HOMO_{nn}(eV)', label_z='Similarity(%)', save_name='results/figure8-1.eps')

    draw(np.array(l_pred), np.array(nn_data)[:,1], np.array(simi),x_range=(-6, -2), y_range=(-6, -2),
         label_x='$LUMO_{gen}(eV)', label_y='LUMO_{nn}(eV)', label_z='Similarity(%)', save_name='results/figure8-2.eps')

    draw(np.array(p_pred), np.array(nn_data)[:, 2], np.array(simi), x_range = (0, 11), y_range = (0, 11),
    label_x = '$PCE_{gen}(eV)', label_y = 'PCE_{nn}(eV)', label_z = 'Similarity(%)', save_name = 'results/figure8-3.eps')


    dic={"SMILES":ms,"HOMO_pred":h_pred,"LUMO_pred":l_pred,"PCE_pred":p_pred,"Similarity":simi,
         "HOMO_nn":np.array(nn_data)[:,0],"LUMO_nn":np.array(nn_data)[:,1],
         "PCE_nn":np.array(nn_data)[:,2],"SMILES_nn":nn_list}
    DataFrame(dic).to_csv('results/Gen.csv')