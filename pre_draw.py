import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn import linear_model
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def pred(n, data, mod):
    tar = np.zeros((len(data),1))
    out = np.zeros((len(data),1))
    model = torch.load(mod, map_location=torch.device('cpu'))
    model.eval()
    for i,(input, target) in enumerate(data):
        output = model(input[1:-1].unsqueeze(0))
        out[i] = output.cpu().detach().item()
        tar[i] = target[n:n + 1].detach().item()
    return tar, out

def draw(x, y, color,range, label_x, label_y, save_name):
    plt.figure()
    plt.scatter(x, y,color=color, s=10)
    plt.xlim(range)
    plt.ylim(range)
    plt.axis("square")
    plt.tick_params(direction='in', labelsize=16)
    plt.xlabel(label_x, {'weight': 'normal', 'size': 20})
    plt.ylabel(label_y, {'weight': 'normal', 'size': 20})
    figure_fig = plt.gcf()
    plt.tight_layout()
    figure_fig.savefig(save_name, format='eps', dpi=300)

if __name__ == "__main__":
    train_data, val_data, test_data = torch.load("data/opv_data.pt")
    target0, output0 = pred(0, test_data, "results/saved_models/50pred.pt")
    target1, output1 = pred(1, test_data, "results/saved_models/51pred.pt")
    target2, output2 = pred(2, test_data, "results/saved_models/54pred.pt")

    draw(target0, output0, 'blue', range=(-8, -5), label_x='$HOMO(eV)$',
         label_y='$HOMO_{pred}(eV)$', save_name='results/figure5-1.eps')

    draw(target1, output1, 'orange', range=(-5, -2), label_x='$LUMO(eV)$',
         label_y='$LUMO_{pred}(eV)$', save_name='results/figure5-2.eps')

    draw(target2, output2, 'c', range=(0, 14), label_x='$PCE(%)$',
         label_y='$PCE_{pred}(%)$', save_name='results/figure5-3.eps')
