# buffer has [q, raw]; q=(mu,sigma), raw=chart data
import numpy as np
import torch


class mybuffer:
    def __init__(self, size:int, batch_size:int):
        self.q = []
        self.raw = []
        self.size = size
        self.batch_size = batch_size

        self.cnt = 0

    
    def put(self, q, raw):
        if len(self.q) < self.size:
            self.q.append(q)
            self.raw.append(raw)
        else:
            self.q[self.cnt] = q
            self.raw[self.cnt] = raw
            
        self.cnt += 1
        if self.cnt > self.size: self.cnt = 0

    
    def get(self, n):
        indices = torch.randint(0, len(self.q), (n // self.batch_size + 1,), dtype=torch.long)
        sampled_q = torch.cat([self.q[i] for i in indices], dim=0)
        sampled_raw = torch.cat([self.raw[i] for i in indices], dim=0)

        return sampled_q[:n], sampled_raw[:n]
    

    def clear(self):
        self.__init__(self.size, self.batch_size)

        