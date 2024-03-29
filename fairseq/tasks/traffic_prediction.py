import os
import torch
from fairseq.tasks import FairseqTask, register_task

from fairseq.data import TrafficDataset

import pandas as pd
import matplotlib.pyplot as plt


import wandb

# python train.py data --task traffic_prediction --arch lstm_traffic --criterion mse_loss --batch-size 16
# C:\Users\rwe180\Documents\python-scripts\pytorch\pyNTF\ 

#python train.py 'data' --task traffic_prediction --criterion mse_loss --arch NTF_traffic --batch-size 8 --optimizer adam --lr 4e-3  --max-tokens 42000 --clip-norm 5.0 --warmup-updates 10 --lr-scheduler inverse_sqrt --update-freq 8

@register_task('traffic_prediction')
class TrafficPredictionTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        # Add some command-line arguments for specifying where the data is
        # located and the maximum supported input length.
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--max-positions', default=1024, type=int,
                            help='max input length')
        parser.add_argument('--segment_lengths_file', default='segment_lengths.txt', type=str,
                            help='file prefix for data')

    @classmethod
    def setup_task(cls, args, **kwargs):
        # Here we can perform any setup required for the task. This may include
        # loading Dictionaries, initializing shared Embedding layers, etc.
        with open(args.segment_lengths_file) as f:
            segment_lengths = f.read().splitlines()
        print('| segment_lengths_file had {} segments'.format(len(segment_lengths)))

        return TrafficPredictionTask(args)

    def __init__(self, args):
        super().__init__(args)
        self.valid_step_num = 0
        # self.segment_lengths = segment_lengths

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        self.seq_len = 10
        self.input_seq_len=1440
        data_file = os.path.join(self.args.data, '{}.csv'.format('jan_mar_30s_pad'))#split))
        self.datasets[split] = TrafficDataset(data_file,seq_len=self.seq_len, train_size=48000,split=split,input_seq_len=1440)
        #if split=='train':
        self.max_vals = self.datasets[split].get_max_vals()

        print('| {} {} {} examples'.format(self.args.data, split, len(self.datasets[split])))
    
    def get_max_vals(self):
        return self.max_vals
    
    def get_seq_len(self):
        return self.seq_len

    def get_input_seq_len(self):
        return self.input_seq_len 

    def max_positions(self):
        """Return the max input length allowed by the task."""
        # The source should be less than *args.max_positions* and the "target"
        # has max length 1.
        return (1e5,1)#(self.args.max_positions*self.seq_len, 1)

    # @property
    # def source_dictionary(self):
    #     """Return the source :class:`~fairseq.data.Dictionary`."""
    #     return None#self.input_vocab

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return None#self.label_vocab

    def valid_step(self, sample, model, criterion):
        model.eval()
        self.valid_step_num += 1
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
            try:
                wandb.log({'valid_loss':loss})
                if self.valid_step_num%100 == 0:
                    net_output = model(**sample['net_input'])
                    print("****")
                    # plt.ion()
                    # plt.pause(0.1)
                    # plt.close('all')
                    # plt.pause(0.1)
                    print(net_output[0].size())
                    
                    preds = net_output[0].view(-1,self.seq_len,90).detach().cpu().numpy()#[0,:,0]#model.get_normalized_probs(net_output, log_probs=True).float()
                    src = sample['net_input']['src_tokens'].view(-1,self.seq_len,90).detach().cpu().numpy()#[0,:,0]# model.get_targets(sample, net_output).float()
                    target = sample['target'].view(-1,self.seq_len,90).detach().cpu().numpy()
                    for i in range(2):
                        for seg in range(0,10):
                            ax = pd.DataFrame(preds[i,:,seg*1]).plot()
                            pd.DataFrame(target[i,:,seg*1]).plot(ax=ax)
                            plt.title(str(i)+"***"+str(seg))
                            plt.pause(0.1)
                            plt.show(block=False)
                            plt.pause(3.0)
                            plt.pause(0.1)
                            try:
                                wandb.log({"chart"+str(i)+"_"+str(seg): plt})
                            except Exception as e:
                                print(e)
                        plt.pause(2.0)
                        plt.close('all')
                    plt.pause(5.0)
                    plt.close('all')
                    # wandb.save('checkpoints/checkpoint_best.pt')
                    # wandb.save('checkpoints/checkpoint_last.pt')
            except Exception as e:
                print(e)
        return loss, sample_size, logging_output

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        wandb.log({'train_loss':loss})
        return loss, sample_size, logging_output

    # We could override this method if we wanted more control over how batches
    # are constructed, but it's not necessary for this tutorial since we can
    # reuse the batching provided by LanguagePairDataset.
    #
    # def get_batch_iterator(
    #     self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
    #     ignore_invalid_inputs=False, required_batch_size_multiple=1,
    #     seed=1, num_shards=1, shard_id=0,
    # ):
    #     (...)