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

import random
import wandb


@register_model('NTF_traffic')
class NTFModel(FairseqEncoderDecoderModel):
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
        max_vals = task.get_max_vals()
        encoder = TrafficNTFEncoder()
        decoder = TrafficNTFDecoder(max_vals=max_vals)
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


class TrafficNTFEncoder(FairseqEncoder):
    """NTF encoder."""
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

        #assert one_sample_length == self.seq_len*one_timestep_size

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

        x_mask = x < 1e-6 #BUG
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

        encoder_padding_mask = x_mask#.sum(dim=2)<1 #BUG: need multidim mask

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

        # x: bsz x output_embed_dim
        x = self.input_proj(input)
        
        # compute attention
        #from fairseq import pdb; pdb.set_trace();
        #[360, 16, 90] * 1, [16, 90]
        attn_scores = (source_hids * x.unsqueeze(0))#.sum(dim=2)
        #[srclen, bsz]

        # attn_scores[attn_scores!=attn_scores] = float('-inf')

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float(-1e6)#float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        #attn_scores = attn_scores.float().masked_fill_(encoder_padding_mask,float('-inf'))
        #attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz
        attn_scores = nn.Softmax(2)(attn_scores)

        #from fairseq import pdb; pdb.set_trace();
        # sum weighted sources
        x = (attn_scores * source_hids).sum(dim=0)
        #x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)
        
        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        # attn_scores [360, 16, 90]
        attn_scores = attn_scores.sum(dim=2)
        return x, attn_scores#.sum(dim=2)


class TrafficNTFDecoder(FairseqIncrementalDecoder):
    """Traffic NTF decoder."""
    def __init__(
        self, hidden_size=90, #input_size=90, output_size=90,
        num_segments = 18,num_var_per_segment = 5, seq_len = 360,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=90, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None, max_vals = None
    ):
        super().__init__(dictionary=None)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=75, profile=None, sci_mode=False)

        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True

        # attention=False
        # self.need_attn = False#True

        self.max_vals = torch.Tensor(max_vals).to(self.device)

        #attention=False
        self.num_segments = num_segments

        self.input_size = self.num_segments * num_var_per_segment
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
                input_size=self.input_size if layer == 0 else hidden_size,#hidden_size + 
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
        
        

        #self.segment_fixed = torch.Tensor([[582.0/1000.,3.],[318.0/1000.,4.],[703.0/1000.,4.],[ 387.0/1000.,4.],[ 300.0/1000.,5.],[ 348.0/1000.,5.],[ 375.0/1000.,4.],[ 300.0/1000.,4.],[ 257.0/1000.,4.],[ 500.0/1000.,4.],[ 484.0/1000.,4.],[ 400.0/1000.,3.],[ 420.0/1000.,3.],[ 589.0/1000.,3.],[ 427./1000.,3.],[ 400.0/1000.,2.],[ 515.0/1000.,2.0],[ 495.0/1000.,3.0]]).to(self.device)#torch.Tensor(self.num_segments, 2)
        self.segment_fixed = torch.Tensor([[0.57237,3.],[0.92267,3.],[0.4238,3.],[0.8092,3.],[1.183,3.],[1.5899,3.],[0.2161,3.],[0.88367,3.],[0.59879,3.],[0.91263,3.],[0.672,3.],[1.7492,3.],[0.28183,3.],[0.85799,3.],[1.0847,3.],[1.1839,3.],[0.56727,3.],[0.5147,3.]]).to(self.device)
        self.model_fixed = torch.Tensor([[10./3600.,17./3600.,23.,1.7,13.]]).to(self.device)

        num_boundry = 3
        num_model_params=8
        self.ntf_vars = num_model_params*self.num_segments+num_boundry
        self.ntf_projection = nn.Linear(self.output_size, self.ntf_vars)

        self.ntf_module = NTF_Module(num_segments=self.num_segments,\
                                     segment_fixed = self.segment_fixed,\
                                     model_fixed = self.model_fixed
                                    )
    
        #self.fc_out = Linear(self.output_size, self.output_size, dropout=dropout_out)


    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
    #def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        encoder_padding_mask = encoder_out['encoder_padding_mask']
        encoder_out = encoder_out['encoder_out']

        if incremental_state is not None:
            print(prev_output_tokens.size())
            # prev_output_tokens = prev_output_tokens[:, -1:]
            prev_output_tokens = prev_output_tokens[:, -1:,:]
            
        # bsz, one_input_size = prev_output_tokens.size()
        # self.seq_len = 360
        # seqlen = self.seq_len
        bsz, seqlen, segment_units = prev_output_tokens.size()

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
            input_feed = x.new_ones(bsz, self.hidden_size) * 0.5 

        attn_scores = x.new_zeros(srclen, seqlen, bsz)#x.new_zeros(segment_units, seqlen, bsz)  #x.new_zeros(srclen, seqlen, bsz)
        outs = []
        
        for j in range(seqlen):
            # from fairseq import pdb; pdb.set_trace()
            # input feeding: concatenate context vector from previous time step
            input_mask = x[j, :, :] > -1e-6
            input_in = (x[j, :, :]*input_mask.float()) + ( (1-input_mask.float())*input_feed)
            #input = torch.clamp(input, min=-1.0, max=1.0)
            #import pdb; pdb.set_trace()

            if random.random() > 0.9999:
                #from fairseq import pdb; pdb.set_trace()
                print(input_in*self.max_vals)
                means = (input_in*(self.max_vals+1e-6)).view(18,5).mean(dim=0).cpu().detach().numpy()
                print("\n\ninput means\t",means)
                wandb.log({"input0": wandb.Histogram(means[0])})
                wandb.log({"input1": wandb.Histogram(means[1])})
                wandb.log({"input2": wandb.Histogram(means[2])})
                wandb.log({"input3": wandb.Histogram(means[3])})
                wandb.log({"input4": wandb.Histogram(means[4])})
                mean_x = x[j, :, :].view(18,5).mean(dim=0)
                print("x[j, :, :] means\t",mean_x.cpu().detach().numpy())
                mean_feed = input_feed.view(18,5).mean(dim=0)
                print("input_feed means\t",mean_feed.cpu().detach().numpy())
            
            # if random.random()>0.0:
            #     input = x[j, :, :]#torch.cat((x[j, :, :], input_feed), dim=1)
            # else:
            #     input = input_feed
            input = input_in
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
            # from fairseq import pdb; pdb.set_trace()
            ntf_input = self.ntf_projection(out)
            boundry_params, segment_params = torch.split(ntf_input, [3,8*self.num_segments],dim=1)
            # boundry_params = torch.Tensor([200.0,10000.0,200.0]).to(self.device)*torch.sigmoid(boundry_params)
            boundry_params = torch.Tensor([200.0,10000.0,200.0]).to(self.device)*torch.sigmoid(boundry_params)
            segment_params = segment_params.view((-1,8,self.num_segments))
            segment_params = torch.cat([torch.sigmoid(segment_params[:,:4,:]),torch.tanh(segment_params[:,4:,:])],dim=1)
                                                            # vf, a, rhocr, g, omegar, omegas, epsq, epsv 
            segment_params =  segment_params* torch.Tensor([[200.0],[2.0],[200.0],[5.0],[100.0],[100.0],[100.0],[10.0]]).to(self.device)
            segment_params = segment_params.permute(0, 2, 1)
            unscaled_input = input_in * self.max_vals

            # print("boundry_params",boundry_params[0,::5].mean().item(),boundry_params.size())
            # print("segment_params",segment_params[0,::5,0].mean().item(),segment_params.size())
            #print(unscaled_input)

            out = self.ntf_module(unscaled_input,segment_params,boundry_params)

            # print(out.mean().item())

            out = out / (self.max_vals+1e-6)

            

            #from fairseq import pdb; pdb.set_trace()

            #out = F.dropout(out, p=self.dropout_out, training=self.training)


            # input feeding
            input_feed = out#.view(-1,360,90)


            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, prev_cells, input_feed),
        )

        # collect outputs across time steps
        #print(torch.stack(outs, dim=0).size())
        # from fairseq import pdb; pdb.set_trace();
        x = torch.stack(outs, dim=1)#.view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        #x = x.transpose(1, 0)


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
        
        #x = x.contiguous().view(bsz,-1)#self.output_size)#self.fc_out(x)
        return x, attn_scores
    
    #my implementation
    def get_normalized_probs(self, net_output, log_probs=None, sample=None):
        return net_output[0]
    
    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        target =  sample['target']
        print(target.size())
        target = target.transpose(1, 2)
        return target
        

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

class NTF_Module(nn.Module):

    def __init__(self, num_segments=18,\
                 segment_fixed = torch.Tensor(18, 2),\
                 model_fixed = torch.Tensor(1, 5)
                ):
        super(NTF_Module, self).__init__()
        
        assert type(num_segments) == int, "num_segments needs to be an int and not "+str(type(num_segments))
        self.num_segments = num_segments

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.segment_fixed = segment_fixed.view((self.num_segments, 2)).to(self.device)
        self.model_fixed = model_fixed.view((1, 5)).to(self.device)
        
        self.T, self.tau, self.nu, self.delta, self.kappa = torch.unbind(model_fixed, dim=1)
        self.Delta, self.lambda_var = torch.unbind(segment_fixed, dim=1)

        self.q_index = 0
        self.rho_index = 1
        self.v_index = 2
        self.r_index = 3
        self.s_index = 4
        
        self.v0_index = 0
        self.q0_index = 1
        self.rhoNp1_index = 2
        
        self.vf_index = 0
        self.a_index = 1
        self.rhocr_index = 2
        self.omegar_index = 4
        self.omegas_index = 5
        self.epsq_index = 6
        self.epsv_index = 7
        self.g_index = 3
        
        self.T_index = 0
        self.tau_index = 1
        self.nu_index = 2
        self.delta_index = 3
        self.kappa_index = 4
        
        self.Delta_index = 0
        self.lambda_index = 1

        self.calculate_velocity = True
        
    
    def future_v(self,x,boundry_params,segment_params):
      TINY = 1e-6
      
      v0, q0, rhoNp1 = torch.unbind(boundry_params, dim=2)
      vf, a_var, rhocr, g, omegar, omegas, epsq, epsv = torch.unbind(segment_params, dim=2)
      #rhocr = torch.clamp(rhocr, min=30, max=500)
      try:
        if random.random() > 0.9999:
            wandb.log({"vf": wandb.Histogram(vf.cpu().detach().numpy())})
            wandb.log({"a_var": wandb.Histogram(a_var.cpu().detach().numpy())})
            wandb.log({"rhocr": wandb.Histogram(rhocr.cpu().detach().numpy())})
            wandb.log({"g": wandb.Histogram(g.cpu().detach().numpy())})
            wandb.log({"omegar": wandb.Histogram(q0.cpu().detach().numpy())})
            wandb.log({"omegas": wandb.Histogram(rhoNp1.cpu().detach().numpy())})
        # tb.add_histogram('vf', vf, epoch)
        # tb.add_histogram('a', a, epoch)
        # tb.add_histogram('rhocr', rhocr, epoch)
        # tb.add_histogram('g', g, epoch)
        # tb.add_histogram('omegar', omegar, epoch)
        # tb.add_histogram('omegas', omegas, epoch)
      except Exception as e:
        print(e)

      current_densities = x[:,:,self.rho_index]
      current_flows = x[:,:,self.q_index]
      if self.calculate_velocity:
        current_velocities = current_flows / (current_densities*self.segment_fixed[:,self.lambda_index]+TINY)
        if random.random() > 0.9999:
            wandb.log({"current_velocities": wandb.Histogram(current_velocities.cpu().detach().numpy())})
            wandb.log({"current_densities": wandb.Histogram(current_densities.cpu().detach().numpy())})
            wandb.log({"current_flows": wandb.Histogram(current_flows.cpu().detach().numpy())})
        current_velocities = torch.clamp(current_velocities, min=5, max=200)
      else:
        current_velocities = x[:,:,self.v_index]
        current_velocities = torch.clamp(current_velocities, min=5, max=200)


      prev_velocities = torch.cat([v0,current_velocities[:,:-1]],dim=1)
      next_densities = torch.cat([current_densities[:,1:],rhoNp1],dim=1)

      stat_speed = vf* torch.exp(torch.div(-1,a_var+TINY)*torch.pow(torch.div(current_densities,rhocr+TINY)+TINY,a_var))
      if random.random() > 0.9999:
        print("stat speed",stat_speed.size(),stat_speed.min().item(),stat_speed.mean().item(),stat_speed.max().item())
        print("v0,q0,rhoNN",v0[0].item(), q0[0].item(), rhoNp1[0].item(),v0.mean().item(), q0.mean().item(), rhoNp1.mean().item())
        print("q1",x[0,0,self.q_index].item(),x[:,0,self.q_index].mean().item())
        print("vf, a, rhocr,g, omegar, omegas, epsq, epsv",vf[0].mean().item(), a_var[0].mean().item(), rhocr[0].mean().item(),g[0].mean().item(), omegar[0].mean().item(), omegas[0].mean().item(), epsq[0].mean().item(), epsv[0].mean().item())
      
      #import pdb; pdb.set_trace()

      return current_velocities + (torch.div(self.T,self.tau+TINY)) * (stat_speed - current_velocities )  \
              + (torch.div(self.T,self.Delta) * current_velocities * (prev_velocities - current_velocities)) \
              - (torch.div(self.nu*self.T, (self.tau*self.Delta)) * torch.div( (next_densities - current_densities), (current_densities+self.kappa)) ) \
              - (torch.div( (self.delta*self.T) , (self.Delta * self.lambda_var) ) * torch.div( (x[:,:,self.r_index]*current_velocities),(current_densities+self.kappa) ) ) \
              + epsv

    def future_rho(self,x,boundry_params):
      v0, q0, rhoNp1 = torch.unbind(boundry_params, dim=2)
      try:
        if random.random() > 0.9999:
            wandb.log({"v0": wandb.Histogram(v0.cpu().detach().numpy())})
            wandb.log({"q0": wandb.Histogram(q0.cpu().detach().numpy())})
            wandb.log({"rhoNp1": wandb.Histogram(rhoNp1.cpu().detach().numpy())})
        # tb.add_histogram('v0', v0, epoch)
        # tb.add_histogram('q0', q0, epoch)
        # tb.add_histogram('rhoNp1', rhoNp1, epoch)
      except Exception as e:
        print(e)
      
      current_flows = x[:,:,self.q_index]
      current_densities = x[:,:,self.rho_index]
      prev_flows = torch.cat([q0,current_flows[:,:-1]],dim=1)
      #import pdb; pdb.set_trace()
      return current_densities + torch.mul(torch.div(self.T,torch.mul(self.Delta,self.lambda_var)),(prev_flows - current_flows + x[:,:,self.r_index] - x[:,:,self.s_index]))
    
    def forward(self,x,segment_params,boundry_params):
      #import pdb; pdb.set_trace()
      
      segment_params = segment_params.view((-1,self.num_segments, 8)) #TODO: remove hardcode
      boundry_params = boundry_params.view((-1,1, 3))  #TODO: remove hardcode
      
      x = x.view(-1,self.num_segments,5) #TODO: remove hardcode
      
    #   invalid_inputs_mask = (x<1e-6).float()
    #   invalid_input_defaults = torch.zeros_like(x).to(self.device)
    #   invalid_input_defaults[:,:,self.v_index] = 100.0
    #   invalid_input_defaults[:,:,self.q_index] = 10000.0
    #   invalid_input_defaults[:,:,self.rho_index] = 20.0

    #   #print("xmean1",x.mean())

    #   x = x + invalid_inputs_mask*invalid_input_defaults

      #print("xmean2",x.mean())
      # 

      
      input_mask = torch.ones_like(x).to(self.device)#, dtype=torch.uint8).to(self.device)
      input_mask[:,:,self.rho_index] = segment_params[:,:,self.g_index]
      
      x = x*input_mask
      x = nn.functional.relu(x)

      future_velocities = self.future_v(x,boundry_params,segment_params)
      future_densities = self.future_rho(x,boundry_params)
      future_occupancies = future_densities / (segment_params[:,:,self.g_index]+1e-6)

      future_flows = future_densities * future_velocities * self.segment_fixed[:,self.lambda_index] + segment_params[:,:,self.epsq_index]

      future_r =  segment_params[:,:,self.omegar_index] + x[:,:,self.r_index]
      future_s = segment_params[:,:,self.omegas_index] + x[:,:,self.s_index]
      
      try:
        if random.random() > 0.9999:
            wandb.log({"future_velocities": wandb.Histogram(future_velocities.cpu().detach().numpy())})
            wandb.log({"future_densities": wandb.Histogram(future_densities.cpu().detach().numpy())})
            wandb.log({"future_occupancies": wandb.Histogram(future_occupancies.cpu().detach().numpy())})
            wandb.log({"future_flows": wandb.Histogram(future_flows.cpu().detach().numpy())})
            wandb.log({"future_r": wandb.Histogram(future_r.cpu().cpu().detach().numpy())})
            # tb.add_histogram('future_velocities', future_velocities, epoch)
            # tb.add_histogram('future_densities', future_densities, epoch)
            # tb.add_histogram('future_occupancies', future_occupancies, epoch)
            # tb.add_histogram('future_flows', future_flows, epoch)
            # tb.add_histogram('future_r', future_r, epoch)
      except Exception as e:
        print(e)

      
      future_velocities = torch.clamp(future_velocities, min=5, max=120)
      future_densities = torch.clamp(future_densities, min=1, max=200)
      future_occupancies = torch.clamp(future_occupancies, min=1, max=30)
      future_flows = torch.clamp(future_flows, min=1, max=10000)
      future_r = torch.clamp(future_r, min=0, max=1000)
      future_s = torch.clamp(future_s, min=0, max=1000)
      
      one_stack =  torch.stack((future_flows,future_occupancies,future_velocities,future_r,future_s),dim=2)

      #import pdb; pdb.set_trace()
      #one_stack[one_stack!=one_stack] =  1.


      return one_stack.view(-1,self.num_segments*5)

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


# @register_model_architecture('NTF', 'NTF')
@register_model_architecture('NTF_traffic', 'NTF_traffic')
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
