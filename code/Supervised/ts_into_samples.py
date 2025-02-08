from hdf5_loader import StockDatasetHDF5
from myconfig import *

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader


def rate(target, flag):
    return (target-flag) / flag * 100

def minmaxscaling(targ:torch.Tensor) -> torch.Tensor:
    tensor_min, _ = torch.min(targ, dim=1, keepdim=True)
    tensor_max, _ = torch.max(targ, dim=1, keepdim=True)
    scaled_tensor = (targ - tensor_min) / (tensor_max - tensor_min)
    tensor_cleaned = torch.nan_to_num(scaled_tensor, nan=0.5, posinf=1e8, neginf=-1e8)
    return tensor_cleaned


class Transforms:
    @staticmethod
    def example(data, hz_patch:dict[str, int]):
        '''
        input shape: dict[str, (df | Any)]; each df ~ (T, 7)
        return shape: dict[str, (abs_tensor, rel_tensor)]; tensor ~ (T-hz_patch[hz]+1, f)
        '''
        result = dict()
        for hz in THZ:
            tensor_data = torch.tensor(data[hz].to_numpy(), dtype=torch.float32)  # (T, 7)
            unfolded = tensor_data.unfold(dimension=0, size=hz_patch[hz], step=1)  # (T, 7, S)
            # (T, p)로 만들어야됨  -> vwap 그 자체 / 평균, 분산, 최대, 최소, 거래량 평균를 넣자. => p = S + 4
            pattern = unfolded[:, 5]
            scaled_pattern = minmaxscaling(pattern)

            pattern_features = list()
            pattern_features.append(torch.std(pattern, axis=1, keepdim=True) * 1000 / torch.mean(pattern, axis=1, keepdim=True))
            pattern_features = torch.cat(pattern_features, dim=1)
            
            nonrel_features = torch.cat([scaled_pattern, pattern_features], dim=1)

            relative_features = list()
            relative_features.append(torch.mean(unfolded[:, 5], axis=1, keepdim=True))
            relative_features.append(torch.min(unfolded[:, 2], axis=1, keepdim=True)[0])
            relative_features.append(torch.max(unfolded[:, 1], axis=1, keepdim=True)[0])
            relative_features = torch.cat(relative_features, dim=1)
            
            # padding
            rel_pad = torch.zeros(size=(hz_patch[hz]-1, relative_features.shape[1]))
            relative_features = torch.cat([rel_pad, relative_features], dim=0)
            nonrel_pad = torch.zeros(size=(hz_patch[hz]-1, nonrel_features.shape[1]))
            nonrel_features = torch.cat([nonrel_pad, nonrel_features], dim=0)
            
            result[hz] = (nonrel_features, relative_features)
        
        return result
    
    @staticmethod
    def yexample(data, label_peroid:int):
        '''
        input shape: tensor ~ (T, 7)
        return shape: tensor ~ (T-label_peroid,)
        '''
        tensor_data = torch.tensor(data.to_numpy(), dtype=torch.float32)  # (T, 7)
        unfolded = tensor_data.unfold(dimension=0, size=label_peroid, step=1)  # (T, 7, label_peroid)
        flag = tensor_data[:unfolded.shape[0]-1, 3].view(-1, 1, 1)
        
        rel = torch.divide(unfolded[1:, :4] - flag, flag) * 100  # t에 대한 t+1 ~ t+lp가 y이므로 1번 인덱스부터 사용됨 
        # average of mean, min, max
        result = torch.div(torch.min(rel[:, 2], dim=1)[0] + torch.max(rel[:, 1], dim=1)[0] + torch.mean(rel[:, 3], dim=1), 3)
        
        return result
    
    @staticmethod
    def get_ymask(data, label_hz, hz_patch, label_peroid, data_min_interval, label_density):
        '''
        input shape: tensor ~ (T, 7) for label_hz
        output shape: tensor ~ (T, 1)
        '''
        # 조건1: data_min_interval보다 긴 데이터 필요
        # 조건2: label_peroid 구간에 대해 label_hz에 데이터 존재
        targ = data['open']
        later_index = targ.index + TO_TIMEDELTA[label_hz](label_peroid)
        locations = targ.index.asof_locs(later_index, np.ones(len(later_index), dtype=bool))
        y_mask = (locations - np.array(range(len(later_index)))) >= label_peroid*label_density
        y_mask = y_mask & (targ.index >= targ.index[0] + data_min_interval) \
            & (targ.index <= targ.index[-(hz_patch[label_hz]+label_peroid)])
        
        y_index = targ.index.astype('int64') // 10**9
        return y_mask, y_index
    
    @staticmethod
    def get_sample(data, x, y, ymask, yind, label_hz, hz_patch, hz_window):
        curidx = {hz:0 for hz in THZ}
        for idx, yt in enumerate(data[label_hz].index):
            # ymask check
            if not ymask[idx]: continue
            
            deb = dict()
            # curidx
            for hz in THZ:
                while True:
                    deb[hz] = data[hz].index[curidx[hz]]
                    if data[hz].index[curidx[hz]] + TO_TIMEDELTA[hz](1) - timedelta(seconds=1) > yt:
                        break
                    else: curidx[hz] += 1
            if curidx[label_hz] == 0: continue
            
            # print(datetime.fromtimestamp(yind[curidx[label_hz]-1] - 60*60*9), deb)
            
            # sample
            x_sample = [dict(), dict()]
            for hz in THZ:
                for rel in [0, 1]:
                    x_st = curidx[hz] - hz_window[hz]
                    if curidx[hz] == 0:
                        # no data -> zeros
                        x_sample[rel][hz] = torch.zeros((hz_window[hz], x[hz][rel].shape[1]))
                        continue
                    #
                    if x_st < 0:
                        # padding
                        pad_size = (0, 0, -x_st, 0)
                        x_sample[rel][hz] = nn.functional.pad(x[hz][rel][:curidx[hz]], pad_size, 'constant', 0)
                    else:
                        # slicing
                        x_sample[rel][hz] = x[hz][rel][x_st:curidx[hz]]
                    #
                    if rel:
                        # relative features
                        flag = data[hz].iat[curidx[hz]-1, 3]
                        x_sample[rel][hz] = torch.divide(x_sample[rel][hz] - flag, flag) * 100
                        x_sample[rel][hz][x_sample[rel][hz] == -100] = 0

            # concat rel & non-rel
            temp = dict()
            for hz in THZ:
                temp[hz] = torch.cat([x_sample[0][hz], x_sample[1][hz]], dim=1)
                
            x_sample = temp
            y_sample = y[curidx[label_hz]-1]
            yield (x_sample, y_sample, yind[curidx[label_hz]-1])
            

class StockDatasetIter(IterableDataset):
    '''
    dataset: 불러온 차트 데이터셋
    hz_window: 각 데이터 샘플에서 참고할 lookback window의 기간(int)
    hz_patch: dataset에서 데이터포인트를 만들 때, patch를 나누는 기간(int)
    label_peroid: label을 만들 때, 데이터를 참고할 기간(int)
    data_min_interval: 최소한의 참고할 기간(delta), less than max(hz_window)
    label_hz: label을 만들 때의 시간 단위(str)
    transform, target_transform: (function)
    condition: 기타 조건
    '''
    def __init__(self, dataset:StockDatasetHDF5, hz_window:dict[str,int], hz_patch:dict[str,int],
                 label_peroid:int, data_min_interval:timedelta, label_hz:str, label_density:float,
                 transform, target_transform, condition=None):
        super().__init__()
        #
        self.dataset = dataset
        self.hz_window = hz_window
        self.label_peroid = label_peroid
        self.data_min_interval = data_min_interval
        self.hz_patch = hz_patch
        self.label_hz = label_hz
        self.label_density = label_density
        self.condition = condition
        self.transform = transform
        self.target_transform = target_transform
        
    def __iter__(self):
        for data in self.dataset:
            x = self.transform(data, self.hz_patch)
            y = self.target_transform(data[self.label_hz], self.label_peroid)
            ymask, yind = Transforms.get_ymask(data[self.label_hz], self.label_hz, self.hz_patch, self.label_peroid, 
                                               self.data_min_interval, self.label_density)
            gen = Transforms.get_sample(data, x, y, ymask, yind, self.label_hz, self.hz_patch, self.hz_window)
            
            print(f"for ticker: {data['ticker']}")
            for result in gen:
                yield result