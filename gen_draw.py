from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import numpy as np
import matplotlib
from utils import process

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def getfp(sm_file):
    with open(sm_file, 'r') as smi:
        smiles = smi.readlines()
    sms = process(smiles)
    ms = [Chem.MolFromSmiles(smile) for smile in smiles]
    fps = []
    for i in range(len(ms)):
        fps += [AllChem.GetMorganFingerprint(ms[i], 2)]
    return sms, fps


if __name__ == "__main__":
    sms0, fps0 = getfp('data/smi_c.txt')
    sms, fps = getfp('results/4sample.txt')

    lth = []
    simi = []
    count = np.random.permutation(len(sms))
    for i in count:
        lth += [len(sms[i])]
        simi += [max([DataStructs.DiceSimilarity(fps[i], fp0) for fp0 in fps0[:20000]]) * 100]

    print(np.mean(simi), np.var(simi), np.std(simi))
    print(np.mean(lth), np.var(lth), np.std(lth))

    plt.figure(0)
    plt.scatter(simi[:10000], lth[:10000])
    plt.ylim((0, 140))
    plt.xticks(np.arange(20, 110, 10))
    plt.yticks(np.arange(0, 160, 20))
    plt.tick_params(direction='in', labelsize=16)
    plt.xlabel('Similarity(%)', {'weight': 'normal', 'size': 20})
    plt.ylabel('Length', {'weight': 'normal', 'size': 20})
    figure_fig = plt.gcf()
    plt.tight_layout()
    figure_fig.savefig('results/figure4-1.eps', format='eps', dpi=300)

