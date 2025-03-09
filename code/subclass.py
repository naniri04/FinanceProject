import sys, os
sys.path.append(os.path.abspath('..'))

from hdf5_loader import StockDatasetHDF5
from myconfig import *
import numpy as np
import torch
from copy import deepcopy
from collections import defaultdict


FEATURE_NUM = 5
LABEL_HORIZON = 5

def rate(target, flag):
    if flag == 0: return target
    target[target == 0] = flag
    return (target-flag) / flag * 100

def get_samples(hdf5_instance:StockDatasetHDF5, hz_dim, targ_hz, tensor=False):
    for j in range(len(hdf5_instance)):
        temp = dict()
        for hz in THZ:
            df = hdf5_instance[j][hz]
            df['timestamp'] = df.index.astype('int64') // 10**9
            # add global feature
            #~
            
            temp[hz] = df.to_numpy()
            
        # make sample
        curidx = {hz:0 for hz in THZ}
        for i in range(len(temp[targ_hz]) - hz_dim[targ_hz]):
            targ_time = temp[targ_hz][i, 7]
            # indexing
            for hz in THZ:
                while True:
                    if temp[hz][curidx[hz], 7] + UNIT_TS[hz] > targ_time + UNIT_TS[targ_hz]: break
                    else: curidx[hz] += 1
                 
            label = dict(); flag = False
            for hz in THZ:
                label[hz] = temp[hz][curidx[hz]:curidx[hz]+LABEL_HORIZON, 3]
                if label[hz].shape[0] < LABEL_HORIZON: flag = True
            if flag: break
            
            abs_chart = dict()
            rel_chart = dict()
            feature = dict()
            info = dict()

            # split / pad
            for hz in THZ:
                if (diff := curidx[hz] - hz_dim[hz]) >= 0:
                    abs_chart[hz] = temp[hz][diff:curidx[hz]]
                else:
                    abs_chart[hz] = np.pad(temp[hz][:curidx[hz]], pad_width=((-diff, 0),(0,0)), mode='constant', constant_values=0)

            info['current_price'] = abs_chart[targ_hz][-1, 3]
            
            # relative prices
            for hz in THZ:
                loc_feats = []
                # 1. relative prices
                for f in range(4):
                    loc_feats.append(rate(abs_chart[hz][:, f], info['current_price']))
                # 2. relative time
                loc_feats.append(((abs_chart[hz][-1, 7] - abs_chart[hz][:, 7]) // UNIT_TS[hz]))
                loc_feats[-1][loc_feats[-1] == abs_chart[hz][-1, 7]//UNIT_TS[hz]] = 0
                
                rel_chart[hz] = np.stack(loc_feats).astype(np.float32)
                
            for hz in THZ:
                label[hz] = rate(label[hz], info['current_price'])
                
            if tensor:
                for hz in THZ:
                    rel_chart[hz] = torch.Tensor(rel_chart[hz])
                    label[hz] = torch.Tensor(label[hz])
            
            yield rel_chart, feature, label, info
            
        yield 0, 0, 0, 0
        

def get_label(labels, label_weight):
    result = torch.zeros((5, labels['1m'].shape[0]))
    for i, hz in enumerate(THZ):
        result[i] = torch.mean(labels[hz], dim=1) * label_weight[hz]
    return torch.sum(result, dim=0)


from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import matplotlib.pyplot as plt
pal = ['r', 'g', 'b', 'c', 'k']

def plot_chart(raws:torch.Tensor, hz_dim, visual_keys:list=None):
    visual_keys = visual_keys if visual_keys else THZ

    fig, ax = plt.subplots(len(raws), 5, figsize=(10, 1*len(raws)))
    fig.tight_layout()
    for i in range(len(raws)):
        for j, hz in enumerate(visual_keys):
            d = raws[i,j,3]
            d[d == 0] = np.nan
            targ_ax = (ax[j] if len(raws)==1 else ax[i, j])
            sns.lineplot(d, ax=targ_ax, c=pal[j])
            targ_ax.set_xlim(0, hz_dim[hz])
            targ_ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def batch_maker(envgen, batch_size:int):
    '''Output shape = (batch, hz, features, seq_len)'''
    rel_charts = [[] for _ in range(5)]; labels = [[] for _ in range(5)]; features = [[] for _ in range(5)]
    infos = []

    for batch_i in range(batch_size):
        rel_chart, feature, label, info = next(envgen)
        if rel_chart:
            for i, hz in enumerate(THZ):
                rel_charts[i].append(rel_chart[hz])
                labels[i].append(label[hz])
                # features[i].append(feature[hz])
            infos.append(info)
        else: continue
        
    for i in range(5): 
        rel_charts[i] = torch.stack(rel_charts[i])
        labels[i] = torch.stack(labels[i])
        # features[i] = torch.stack(features[i])

    rel_charts = torch.stack(rel_charts, axis=1)
    labels = torch.stack(labels, axis=1)
    # features = torch.stack(features, axis=1)

    return rel_charts, features, labels, infos