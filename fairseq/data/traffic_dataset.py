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

from . import FairseqDataset

#TODO: Delete train_size

class TrafficDataset(FairseqDataset):
    def __init__(self, csv_file, seq_len=360, train_size=48000,vol_multiple = 1.0,
                scale_input = True, scale_output = False,
                shuffle=True, input_feeding=True,
                max_sample_size=None, min_sample_size=None
                ):
        super().__init__()
        

        self.train_size = train_size
        self.all_data = pd.read_csv(csv_file,index_col=0)
        self.all_data.iloc[:,::5] = self.all_data.iloc[:,::5] * vol_multiple
        
        self.scale_input = scale_input
        self.scale_output = scale_output
    
        self.max_vals = self.all_data.iloc[:self.train_size,:].max().values+1.0
        self.seq_len = seq_len

        self.input_feeding = input_feeding

        self.max_sample_size = max_sample_size if max_sample_size is not None else sys.maxsize
        self.min_sample_size = min_sample_size if min_sample_size is not None else self.max_sample_size

        self.shuffle = shuffle

    def __getitem__(self, index):

        input_len = self.seq_len
        label_len = self.seq_len

        NEG = -1e-3

        one_input = self.all_data.iloc[index:index+self.seq_len, :].values
        if self.scale_input:
          one_input = one_input/self.max_vals
        one_input = np.reshape(one_input,-1)
        
        one_label = self.all_data.iloc[index+self.seq_len:index+self.seq_len+label_len, :].values
        if self.scale_output:
          one_label = one_label/self.max_vals
        one_label = np.reshape(one_label,-1)

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
        return len(self.all_data) // self.seq_len - 1

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens = torch.LongTensor([s['source'] for s in samples])
        src_lengths = torch.LongTensor([len(s['source']) for s in samples])

        prev_output_tokens = None
        target = None
        if samples[0].get('target', None) is not None:
            target = torch.LongTensor([s['target'] for s in samples])
            ntokens = sum(len(s['target']) for s in samples)

            if self.input_feeding:
                # we create a shifted version of targets for feeding the
                # previous output token(s) into the next decoder step
                # previous_output = itertools.chain([samples[-1]['source']],[s['target'] for s in samples[:-1]])
                previous_output = [samples[-1]['source']] + [s['target'] for s in samples[:-1]]
                prev_output_tokens = torch.LongTensor(previous_output)
                # prev_output_tokens = merge(
                #     'target',
                #     left_pad=left_pad_target,
                #     move_eos_to_beginning=True,
                # )
                # prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
        else:
            ntokens = sum(len(s['source']) for s in samples)
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