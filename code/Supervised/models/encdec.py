import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import subclass as sc

class CustomAddLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CustomAddLayer, self).__init__()
        self.affine = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
        
    def forward(self, cur, next):
        cur = self.affine(cur)
        cur = torch.add(cur, next)
        cur = self.relu(cur)
        
        return cur


class CustomCNN(nn.Module):
    def __init__(self, hz_dim:dict, hz_order:list, out_dim:int):
        super(CustomCNN, self).__init__()
        
        def get_conv1d_output_length(conv_layer:nn.Conv1d, input_length):
            return (input_length + 2 * conv_layer.padding[0] 
                    - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) 
                    - 1) // conv_layer.stride[0] + 1
            
        self.hzs = hz_order
        self.out_dim = out_dim
        extractors = {}; hz_latent_dim = {}
        
        #region extractors
        for hz, dim in hz_dim.items():
            feat_dim = sc.FEATURE_NUM
            layers = []
            for layer_num in range(5):
                layers.append((f"{hz}_depth_conv{layer_num}", 
                               nn.Conv1d(feat_dim, feat_dim, groups=feat_dim, kernel_size=5, stride=2, padding=2)))
                # layers.append((f"{hz}_batch_norm{layer_num}-1", nn.BatchNorm1d(feat_dim)))
                layers.append((f"{hz}_relu{layer_num}-1", nn.ReLU()))
                # feat_dim *= 2
                
                layers.append((f"{hz}_point_conv{layer_num}", nn.Conv1d(feat_dim, feat_dim*2, kernel_size=1)))
                # layers.append((f"{hz}_batch_norm{layer_num}-2", nn.BatchNorm1d(feat_dim*2)))
                layers.append((f"{hz}_relu{layer_num}-2", nn.ReLU()))
                
                feat_dim *= 2
                
            layers.append((f"{hz}_avg_pool", nn.AdaptiveAvgPool1d(1)))
            layers.append((f"{hz}_flatten", nn.Flatten()))
            layers = OrderedDict(layers)
            
            extractors[hz] = nn.Sequential(layers)  # -> (latent_dim)
            hz_latent_dim[hz] = feat_dim
        #regionend

        self.extractors = nn.ModuleDict(extractors)
        
        self.mergers = dict()
        for hz_idx in range(len(self.hzs)-1):
            cur_hz, next_hz = self.hzs[hz_idx], self.hzs[hz_idx+1]
            self.mergers[cur_hz] = CustomAddLayer(hz_latent_dim[cur_hz], hz_latent_dim[next_hz])
        self.mergers = nn.ModuleDict(self.mergers)

        # Update the features dim manually
        self.features_dim = hz_latent_dim[next_hz]
        
        self.mu_layer = nn.Sequential(
            nn.Linear(self.features_dim, self.features_dim),
            nn.Linear(self.features_dim, self.out_dim),
        )
        self.sigma_layer = nn.Sequential(
            nn.Linear(self.features_dim, self.features_dim),
            nn.Linear(self.features_dim, self.out_dim),
        )
        

    def forward(self, observations):        
        latent_tensors = dict()
        for i, hz in enumerate(self.hzs):
            latent_tensors[hz] = self.extractors[hz](observations[:,i])
        
        latent_state = latent_tensors[self.hzs[0]]
        for hz_idx in range(len(self.hzs)-1):
            latent_state = self.mergers[self.hzs[hz_idx]](latent_state, latent_tensors[self.hzs[hz_idx+1]])
        
        mu = self.mu_layer(latent_state)
        sigma = self.sigma_layer(latent_state)
        return torch.stack((mu, sigma), dim=1)