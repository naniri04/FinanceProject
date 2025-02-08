import sys, os
sys.path.append(os.path.abspath('..'))

from hdf5_loader import StockDatasetHDF5
from myconfig import *
import numpy as np
import torch


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
            
            # split / pad
            result = dict()
            for hz in THZ:
                if (diff := curidx[hz] - hz_dim[hz]) >= 0:
                    result[hz] = temp[hz][diff:curidx[hz]]
                else:
                    result[hz] = np.pad(temp[hz][:curidx[hz]], pad_width=((-diff, 0),(0,0)), mode='constant', constant_values=0)
                    
            result['current_price'] = result[targ_hz][-1, 3]
            
            # add local feature
            for hz in THZ:
                loc_feats = []
                # 1. relative prices
                for f in range(4):
                    loc_feats.append(rate(result[hz][:, f], result['current_price']))
                # 2. relative time
                loc_feats.append(((result[hz][-1, 7] - result[hz][:, 7]) // UNIT_TS[hz]))
                loc_feats[-1][loc_feats[-1] == result[hz][-1, 7]//UNIT_TS[hz]] = 0
                
                result[hz] = np.stack(loc_feats).astype(np.float32)
                
            for hz in THZ:
                label[hz] = rate(label[hz], result['current_price'])
                
            if tensor:
                for hz in THZ:
                    result[hz] = torch.Tensor(result[hz])
                    label[hz] = torch.Tensor(label[hz])
            
            yield result, label
            
        yield 0, 0
        

def get_label(labels, label_weight):
    result = torch.zeros((5, labels['1m'].shape[0]))
    for i, hz in enumerate(THZ):
        result[i] = torch.mean(labels[hz], dim=1) * label_weight[hz]
    return torch.sum(result, dim=0)