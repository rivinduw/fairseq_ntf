# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import numpy as np
import sys
import torch
import torch.nn.functional as F

import pandas as pd
import itertools

import wandb

from . import FairseqDataset

from itertools import cycle, islice

#TODO: Delete train_size

class TrafficDataset(FairseqDataset):
    def __init__(self, csv_file, seq_len=360, train_size=48000,vol_multiple = 1.0,
                scale_input = True, scale_output = True,
                shuffle=True, input_feeding=True,
                max_sample_size=None, min_sample_size=None,split='train'
                ):
        super().__init__()
        
        #train_size = 360*16
        valid_size = train_size
        self.train_size = train_size
        
        self.all_data = pd.read_csv(csv_file,index_col=0)
        print("##Length of Dataset: ",len(self.all_data))
        self.all_data.iloc[:,::5] = self.all_data.iloc[:,::5] * vol_multiple

        self.all_data_pad = pd.read_csv(csv_file.replace('.csv','_pad.csv'),index_col=0)
        self.all_data_pad.iloc[:,::5] = self.all_data_pad.iloc[:,::5] * vol_multiple

        self.seq_len = seq_len
        # self.train_size = len(self.all_data)//3

        self.max_vals = self.all_data.iloc[:self.train_size,:].max().values+1.0
        self.max_vals[0::5] = 10000.0
        self.max_vals[1::5] = 100.0
        self.max_vals[2::5] = 100.0
        self.max_vals[3::5] = 1000.0
        self.max_vals[4::5] = 1000.0
        print(self.max_vals)

        if split=='train':
            self.all_data = self.all_data.iloc[:self.train_size,:]
            self.all_data_pad = self.all_data_pad .iloc[:self.train_size,:]
            print("###Length of Dataset: ",len(self.all_data))
        elif split=='valid':
            print("valid SET")
            self.all_data = self.all_data.iloc[self.train_size:self.train_size+valid_size,:]
            self.all_data_pad = self.all_data_pad.iloc[self.train_size:self.train_size+valid_size,:]
            print("###Length of Dataset: ",len(self.all_data))
        else:
            self.all_data = self.all_data.iloc[self.train_size+valid_size:,:]
            self.all_data_pad = self.all_data_pad.iloc[self.train_size+valid_size:,:]
        
        self.scale_input = scale_input
        self.scale_output = scale_output

        self.input_feeding = input_feeding

        self.max_sample_size = max_sample_size if max_sample_size is not None else sys.maxsize
        self.min_sample_size = min_sample_size if min_sample_size is not None else self.max_sample_size

        self.shuffle = shuffle
    
    def get_max_vals(self):
        return self.max_vals
    
    def __getitem__(self, index):

        input_len = self.seq_len
        label_len = self.seq_len

        NEG = -1e-3

        one_input = self.all_data_pad.iloc[index:index+self.seq_len, :].values
        if self.scale_input:
          one_input = one_input/self.max_vals
        one_input = np.reshape(one_input,-1)
        
        one_label = self.all_data.iloc[index+self.seq_len:index+self.seq_len+label_len, :].values
        if self.scale_output:
          one_label = one_label/self.max_vals
        #one_label = np.reshape(one_label,-1)

        # print(one_label.size())
        # from fairseq import pdb; pdb.set_trace()
        return {
            'id': index,
            'source': one_input.astype('float'),
            'target': one_label.astype('float'),
        }

    def resample(self, x, factor):
        return F.interpolate(x.view(1, 1, -1), scale_factor=factor).squeeze()

    def __len__(self):
        return len(self.all_data) // self.seq_len #- self.seq_len# - 1 #- 4* self.seq_len# - 2 * self.seq_len - 1

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens = torch.FloatTensor([s['source'] for s in samples])
        src_lengths = torch.LongTensor([len(s['source']) for s in samples])

        target = torch.FloatTensor([s['target'] for s in samples])
        ntokens = sum(len(s['target']) for s in samples)
        #max_list = list(islice(cycle(self.max_vals), 32400))
        #from fairseq import pdb; pdb.set_trace();
        previous_output = [s['target'][:-1] for s in samples] # [samples[0]['target'][0]]
        previous_output[0] = np.insert(previous_output[0],0,samples[0]['source'][-1],axis=0)
        #from fairseq import pdb; pdb.set_trace();
        # previous_output = [samples[0]['target']] + [s['target'] for s in samples[:-1]] #BUG: not quite right
        prev_output_tokens = torch.FloatTensor(previous_output)

        # prev_output_tokens = None
        # target = None
        # if samples[0].get('target', None) is not None:
        #     target = torch.FloatTensor([s['target'] for s in samples])
        #     ntokens = sum(len(s['target']) for s in samples)

        #     if self.input_feeding:
        #         # we create a shifted version of targets for feeding the
        #         # previous output token(s) into the next decoder step
        #         # previous_output = itertools.chain([samples[-1]['source']],[s['target'] for s in samples[:-1]])
        #         previous_output = [self.max_vals*samples[-1]['source']] + [s['target'] for s in samples[:-1]]
        #         prev_output_tokens = torch.FloatTensor(previous_output)
        #         # prev_output_tokens = merge(
        #         #     'target',
        #         #     left_pad=left_pad_target,
        #         #     move_eos_to_beginning=True,
        #         # )
        #         # prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
        # else:
        #     ntokens = sum(len(s['source']) for s in samples)
        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            },
            'target': target,
        }
        # batch = {
        #     'id': id,
        #     'net_input': {
        #         'sources': sources,
        #         'src_lens': torch.ones(sources.size())
        #     },
        #     'target': target,
        # }
        if prev_output_tokens is not None:
            batch['net_input']['prev_output_tokens'] = prev_output_tokens
        # from fairseq import pdb; pdb.set_trace()
        return batch

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return len(self.all_data.iloc[index:index+self.seq_len, :].values.reshape(-1))
    

    # def ordered_indices(self):
    #     """Return an ordered list of indices. Batches will be constructed based
    #     on this order."""

    #     if self.shuffle:
    #         order = [np.random.permutation(len(self))]
    #     else:
    #         order = [np.arange(len(self))]

    #     order.append(self.seq_len*np.ones(len(self)))#self.sizes)
    #     from fairseq import pdb; pdb.set_trace()
    #     return np.lexsort(order)
