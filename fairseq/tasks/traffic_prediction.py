import os
import torch
from fairseq.tasks import FairseqTask, register_task

from fairseq.data import TrafficDataset

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

        return TrafficPredictionTask(args,segment_lengths)

    def __init__(self, args, segment_lengths):
        super().__init__(args)
        self.segment_lengths = segment_lengths

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        data_file = os.path.join(self.args.data, '{}.csv'.format('fakedata'))#split))
        self.datasets[split] = TrafficDataset(data_file,seq_len=360, train_size=4800,vol_multiple = 120.0)
        print('| {} {} {} examples'.format(self.args.data, split, len(self.datasets[split])))
        

    def max_positions(self):
        """Return the max input length allowed by the task."""
        # The source should be less than *args.max_positions* and the "target"
        # has max length 1.
        return (self.args.max_positions, 1)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return None#self.input_vocab

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return None#self.label_vocab

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