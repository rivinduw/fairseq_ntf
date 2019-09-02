# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import AdaptiveSoftmax


@register_model('lstm_traffic')
class LSTMModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
    
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        encoder = TrafficLSTMEncoder()
        decoder = TrafficLSTMDecoder()
        return cls(encoder, decoder)
    
    # def forward(self, src_tokens,  prev_output_tokens, **kwargs):#src_lengths,
    #     """
    #     Run the forward pass for an encoder-decoder model.

    #     First feed a batch of source tokens through the encoder. Then, feed the
    #     encoder output and previous decoder outputs (i.e., teacher forcing) to
    #     the decoder to produce the next outputs::

    #         encoder_out = self.encoder(src_tokens, src_lengths)
    #         return self.decoder(prev_output_tokens, encoder_out)

    #     Args:
    #         src_tokens (LongTensor): tokens in the source language of shape
    #             `(batch, src_len)`
    #         src_lengths (LongTensor): source sentence lengths of shape `(batch)`
    #         prev_output_tokens (LongTensor): previous decoder outputs of shape
    #             `(batch, tgt_len)`, for teacher forcing

    #     Returns:
    #         tuple:
    #             - the decoder's output of shape `(batch, tgt_len, vocab)`
    #             - a dictionary with any model-specific outputs
    #     """
    #     encoder_out = self.encoder(src_tokens, **kwargs) # src_lengths=src_lengths,
    #     decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        
    #     # negatives = self.sample_negatives(decoder_out)
    #     # y = decoder_out.unsqueeze(0)
    #     # targets = torch.cat([y, negatives], dim=0)
    #     print(decoder_out.size())
        
    #     return decoder_out


class TrafficLSTMEncoder(FairseqEncoder):
    """LSTM encoder."""
    def __init__(
        self, hidden_size=512, num_layers=1, #input_size=90,
        seq_len = 360,num_segments = 18,num_var_per_segment = 5,
        dropout_in=0.1, dropout_out=0.1, bidirectional=False,padding_value=0):
        super().__init__(dictionary=None)
        
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.seq_len = seq_len
        self.num_segments = num_segments
        self.num_var_per_segment = num_var_per_segment

        self.input_size = num_segments * num_var_per_segment
        self.output_units = self.input_size

        self.lstm = LSTM(
            input_size=self.input_size,
            hidden_size=self.output_units,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )

        self.padding_value = padding_value
        # from fairseq import pdb; pdb.set_trace()

        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):#def forward(self, input_x):#input_x,
        
        input_x = src_tokens
        # from fairseq import pdb; pdb.set_trace()
        bsz, one_sample_length = input_x.size()
        
        one_timestep_size = self.num_segments*self.num_var_per_segment

        assert one_sample_length == self.seq_len*one_timestep_size

        x = input_x.view(-1,self.seq_len,one_timestep_size).float()
        #print("x_mean volume",x[:,:,::5].mean())
        #print("x_mean occupancy",x[:,:,1::5].mean())
        #print("x_mean speed",x[:,:,2::5].mean())
        #print("x_mean rin",x[:,:,3::5].mean())
        #print("x_mean rout",x[:,:,4::5].mean())
        
        # x = F.dropout(x, p=self.dropout_in, training=self.training)
        # print("x2",x.size())

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # print("x3",x.size())

        x_mask = x > 1e-6 #BUG
        # print("x_mask",x_mask.size())

        # apply LSTM
        if self.bidirectional:
            #state_size = 2 * self.num_layers, bsz, self.hidden_size
            state_size = 2 * self.num_layers, bsz, one_timestep_size
        else:
            # state_size = self.num_layers, bsz, self.hidden_size
            state_size = self.num_layers, bsz, one_timestep_size
        
        
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        lstm_outs, (final_hiddens, final_cells) = self.lstm(x, (h0, c0))

        # print("lstm_outs",lstm_outs.size())

        x = lstm_outs

        # unpack outputs and apply dropout
        # x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_value)
        # x = self.output_projection(lstm_outs)
        # x = F.dropout(x, p=self.dropout_out, training=self.training)
        # assert list(x.size()) == [seqlen, bsz, self.output_units]
        # assert list(x.size()) == [seqlen, bsz, segment_units]

        if self.bidirectional:

            def combine_bidir(outs):
                out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
                return out.view(self.num_layers, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = x_mask.sum(dim=2)<1 #BUG: need multidim mask

        # print("x",x.size())
        # print("final_hiddens",final_hiddens.size())
        # print("final_cells",final_cells.size())

        # from fairseq import pdb; pdb.set_trace()

        # return {
        #     'encoder_out': (x, final_hiddens, final_cells),
        #     'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        # }
        return {
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(1, new_order)
            for eo in encoder_out['encoder_out']
        )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim
        # import fairseq.pdb as pdb; pdb.set_trace()

        # x: bsz x output_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


class TrafficLSTMDecoder(FairseqIncrementalDecoder):
    """Traffic LSTM decoder."""
    def __init__(
        self, hidden_size=90, #input_size=90, output_size=90,
        num_segments = 18,num_var_per_segment = 5, seq_len = 360,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=90, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None,
    ):
        super().__init__(dictionary=None)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True

        # attention=False
        # self.need_attn = False#True

        attention=False

        self.input_size = num_segments * num_var_per_segment
        self.output_size = self.input_size
        self.seq_len = seq_len

        self.adaptive_softmax = None

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size:
            self.encoder_hidden_proj = Linear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = Linear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=hidden_size + self.input_size if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])
        if attention:
            # TODO make bias configurable
            self.attention = AttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=False)
        else:
            self.attention = None
        if hidden_size != self.output_size:
            self.additional_fc = Linear(hidden_size, self.output_size)
        # if adaptive_softmax_cutoff is not None:
        #     # setting adaptive_softmax dropout to dropout_out for now but can be redefined
        #     self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, hidden_size, adaptive_softmax_cutoff,
        #                                             dropout=dropout_out)
        # elif not self.share_input_output_embed:
            # self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)
        self.fc_out = Linear(self.output_size, self.output_size, dropout=dropout_out)

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
    #def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        encoder_padding_mask = encoder_out['encoder_padding_mask']
        encoder_out = encoder_out['encoder_out']

        if incremental_state is not None:
            # prev_output_tokens = prev_output_tokens[:, -1:]
            prev_output_tokens = prev_output_tokens[:, -1:,:]
        bsz, one_input_size = prev_output_tokens.size()
        # self.seq_len = 360
        seqlen = self.seq_len
        # bsz, seqlen, segment_units = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
        srclen = encoder_outs.size(0)

        # embed tokens
        # x = self.embed_tokens(prev_output_tokens)
        
        x = prev_output_tokens.view(-1,self.seq_len,self.input_size).float()
        # print(x.size())
        # x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)

        attn_scores = x.new_zeros(srclen, seqlen, bsz)
        outs = []
        
        for j in range(seqlen):
            #from fairseq import pdb; pdb.set_trace()
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], input_feed), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs, encoder_padding_mask)
            else:
                out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, prev_cells, input_feed),
        )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None

        # project back to size of vocabulary
        if self.adaptive_softmax is None:
            if hasattr(self, 'additional_fc'):
                x = self.additional_fc(x)
                x = F.dropout(x, p=self.dropout_out, training=self.training)
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            # else:
            #     x = self.fc_out(x)
        # import fairseq.pdb as pdb; pdb.set_trace()#[:,-1,:]
        x = self.fc_out(x)
        return x, attn_scores
    
    #my implementation
    def get_normalized_probs(self, net_output, log_probs=None, sample=None):
        return net_output[0]

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    # def make_generation_fast_(self, need_attn=False, **kwargs):
    #     self.need_attn = need_attn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


# @register_model_architecture('lstm', 'lstm')
@register_model_architecture('lstm_traffic', 'lstm_traffic')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_freeze_embed = getattr(args, 'encoder_freeze_embed', False)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', args.encoder_embed_dim)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', False)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', args.dropout)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', args.dropout)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_freeze_embed = getattr(args, 'decoder_freeze_embed', False)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', args.decoder_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.decoder_attention = getattr(args, 'decoder_attention', '1')
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', args.dropout)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '10000,50000,200000')
