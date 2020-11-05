from typing import Dict, Any, List, Tuple, Union
import os
import luigi
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchbearer import Trial
from mars_gym.meta_config import ProjectConfig
from mars_gym.model.abstract import RecommenderModule
from mars_gym.model.bandit import BanditPolicy
from mars_gym.torch.init import lecun_normal_init
import pickle

from numpy.random.mtrand import RandomState
import random

from typing import Optional, Callable, Tuple
import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util.transformer import *
import copy


#---------------------------------- AUX --------------------

def load_embedding(_n_items, n_factors, path_item_embedding, path_from_index_mapping, index_mapping, freeze_embedding):

    if path_from_index_mapping and path_item_embedding:
        embs = nn.Embedding(_n_items, n_factors)

        # Load extern index mapping
        if os.path.exists(path_from_index_mapping):
            with open(path_from_index_mapping, "rb") as f:
                from_index_mapping = pickle.load(f)    

        # Load weights embs
        extern_weights = np.loadtxt(path_item_embedding)#*100
        intern_weights = embs.weight.detach().numpy()

        # new_x =  g(f-1(x))
        reverse_index_mapping = {value_: key_ for key_, value_ in index_mapping['ItemID'].items()}
        
        embs_weights = np.array([extern_weights[from_index_mapping['ItemID'][reverse_index_mapping[i]]] if from_index_mapping['ItemID'][reverse_index_mapping[i]] > 1 else intern_weights[i] for i in np.arange(_n_items) ])
        #from_index_mapping['ItemID'][reverse_index_mapping[i]]
        #from IPython import embed; embed()
        embs = nn.Embedding.from_pretrained(torch.from_numpy(embs_weights).float(), freeze=freeze_embedding)
        
    elif path_item_embedding:
        # Load extern embs
        extern_weights = torch.from_numpy(np.loadtxt(path_item_embedding)).float()
    
        embs = nn.Embedding.from_pretrained(extern_weights, freeze=freeze_embedding)
    else:
        embs = nn.Embedding(_n_items, n_factors)

    return embs

class LinearWeightedAvg(nn.Module):
    def __init__(self, n_inputs):
        super(LinearWeightedAvg, self).__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(1)) for i in range(n_inputs)])

    def forward(self, input):
        res = 0
        for emb_idx, emb in enumerate(input):
            res += emb * self.weights[emb_idx]
        return res
        
class EmbeddingDropout(nn.Module):
    def __init__(self, p: float = 0.5,) -> None:
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p

    # def forward(self, *embs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    #     if self.training and self.p:
    #         mask = {}
    #         for emb in embs:
    #             mask[emb] = torch.bernoulli(
    #                 torch.tensor(1 - self.p, device=embs[0].device).expand(
    #                     *embs[0].shape
    #                 )
    #             ) / (1 - self.p)
            
    #         return tuple(emb * mask[emb] for emb in embs)
    #     return tuple(embs)

    def forward(self, *embs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if self.training and self.p:
            mask = {}
            mask = torch.bernoulli(
                torch.tensor(1 - self.p, device=embs[0].device).expand(
                    *embs[0].shape
                )
            ) / (1 - self.p)
            
            return tuple(emb * mask for emb in embs)
        return tuple(embs)

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# --------------------------


class DotModel(RecommenderModule):
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int,
        path_item_embedding: str,
        from_index_mapping: str,
        dropout: float,
        hist_size: int,
        freeze_embedding: bool
    ):
        super().__init__(project_config, index_mapping)
        self.path_item_embedding = path_item_embedding

        self.item_embeddings = load_embedding(self._n_items, n_factors, path_item_embedding, 
                                                from_index_mapping, index_mapping, freeze_embedding)

        self.dense = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hist_size, int(hist_size / 2)),
            nn.SELU(),
            nn.Linear(int(hist_size / 2), 1),
        )
        self.use_normalize = True
        self.weight_init = lecun_normal_init
        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)
            
    def flatten(self, input):
        return input.view(input.size(0), -1)

    def normalize(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if self.use_normalize:
            x = F.normalize(x, p=2, dim=dim)
        return x

    def forward(self, session_ids, item_ids, item_history_ids):
        # Item emb
        item_emb = self.normalize(self.item_embeddings(item_ids))

        # Item History embs
        hist_emb = self.normalize(self.item_embeddings(item_history_ids))
        
        dot_prod = torch.matmul(item_emb.unsqueeze(1), hist_emb.permute(0, 2, 1))

        mask_2 = (item_history_ids != 0).unsqueeze(-2).to(item_history_ids.device).float()
        output = dot_prod * mask_2
        # output = output.sum(2)/(mask_2.sum(2) + 0.1) # normalize
        # out = torch.sigmoid(output)

        dot_prod = self.flatten(output)
        out = torch.sigmoid(self.dense(dot_prod))

        return out

class TransformerModel(RecommenderModule):
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int,
        path_item_embedding: str,
        from_index_mapping: str,
        dropout: float,
        hist_size: int,
        freeze_embedding: bool
    ):
        super().__init__(project_config, index_mapping)
        self.path_item_embedding = path_item_embedding
        
        ninp = 100
        nhid = 100
        nhead = 1
        nlayers = 1

        self.item_embeddings = load_embedding(self._n_items, n_factors, path_item_embedding, 
                                                from_index_mapping, index_mapping, freeze_embedding)


        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers =  nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder =  nn.TransformerEncoder(encoder_layers, nlayers)


        self.transform_heads = 1
        self.transform_n = 1
        self.num_filters = 32
        self.filter_sizes: List[int] = [1, 3, 5]
        self.conv_size_out = len(self.filter_sizes) * self.num_filters


        self.pe = PositionalEncoder(n_factors)
        self.layers = self.get_clones(EncoderLayer(
            n_factors, self.transform_heads, dropout=dropout), self.transform_n)
        self.norm = Norm(n_factors)

        self.convs1 = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (K, n_factors)) for K in self.filter_sizes])


        self.dense = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hist_size+self.conv_size_out, int(hist_size / 2)),
            nn.SELU(),
            nn.Linear(int(hist_size / 2), 1),
        )

        self.use_normalize = True
        self.weight_init = lecun_normal_init
        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)
    
    #We can then build a convenient cloning function that can generate multiple layers:
    def get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
            
    def layer_transform(self, x, mask=None):
        for i in range(self.transform_n):
            x = self.layers[i](x, mask)
        x = self.norm(x)
        return x            

    def conv_block(self, x):
	# conv_out.size() = (batch_size, out_channels, dim, 1)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)

        return x

    def flatten(self, input):
        return input.view(input.size(0), -1)

    def normalize(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if self.use_normalize:
            x = F.normalize(x, p=2, dim=dim)
        return x
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, session_ids, item_ids, item_history_ids):
        # Item emb
        item_emb = self.normalize(self.item_embeddings(item_ids))

        # Item History embs
        hist_emb = self.normalize(self.item_embeddings(item_history_ids))

        # Create transform mask
        #mask = self._generate_square_subsequent_mask(len(item_history_ids)).to(item_history_ids.device)

        #src = self.pos_encoder(hist_emb)
        #hist_emb = self.transformer_encoder(src, mask)

        mask = (item_history_ids != 0).unsqueeze(-2)
        hist_emb = self.pe(hist_emb)
        hist_emb = self.layer_transform(hist_emb, mask)        


        hist_conv = self.conv_block(hist_emb)

        # 
        mask_2 = (item_history_ids != 0).unsqueeze(-2).float()
        dot_prod = torch.matmul(item_emb.unsqueeze(1), hist_emb.permute(0, 2, 1))
        output = dot_prod * mask_2

        dot_prod = self.flatten(output)
        out = torch.sigmoid(self.dense(torch.cat((dot_prod, hist_conv), dim=1)))

        return out

class Caser(RecommenderModule):
    '''
    https://github.com/graytowne/caser_pytorch
    https://arxiv.org/pdf/1809.07426v1.pdf
    '''
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        path_item_embedding: str,
        from_index_mapping: str,
        freeze_embedding: bool,
        n_factors: int,
        p_L: int,
        p_d: int,
        p_nh: int,
        p_nv: int,
        dropout: float,
        hist_size: int,
    ):
        super().__init__(project_config, index_mapping)
        self.path_item_embedding = path_item_embedding
        self.hist_size = hist_size
        self.n_factors = n_factors
        
        # init args
        L = p_L
        dims = p_d
        self.n_h = p_nh
        self.n_v = p_nv
        self.drop_ratio = dropout
        self.ac_conv = F.relu#activation_getter[p_ac_conv]
        self.ac_fc = F.relu#activation_getter[p_ac_fc]
        num_items = self._n_items
        dims = n_factors

        # user and item embeddings
        #self.user_embeddings = nn.Embedding(num_users, dims)
        self.item_embeddings = load_embedding(self._n_items, n_factors, path_item_embedding, 
                                                from_index_mapping, index_mapping, freeze_embedding)


        # vertical conv layer
        self.conv_v = nn.Conv2d(1, self.n_v, (L, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(L)]
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, dims)) for i in lengths])

        # fully-connected layer
        self.fc1_dim_v = self.n_v * dims
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        # W1, b1 can be encoded with nn.Linear
        self.fc1 = nn.Linear(fc1_dim_in, dims)
        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(num_items, dims) #+dims
        self.b2 = nn.Embedding(num_items, 1)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        #self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

        self.cache_x = None


    def forward(self, session_ids, item_ids, item_history_ids): # for training        
        """
        The forward propagation used to get recommendation scores, given
        triplet (user, sequence, targets).
        Parameters
        ----------
        item_history_ids: torch.FloatTensor with size [batch_size, max_sequence_length]
            a batch of sequence
        user_var: torch.LongTensor with size [batch_size]
            a batch of user
        item_ids: torch.LongTensor with size [batch_size]
            a batch of items
        for_pred: boolean, optional
            Train or Prediction. Set to True when evaluation.
        """

        # Embedding Look-up
        item_embs = self.item_embeddings(item_history_ids).unsqueeze(1)  # use unsqueeze() to get 4-D
        #user_emb = self.user_embeddings(user_var).squeeze(1)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_embs)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_embs).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)

        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        x = z #torch.cat([z, user_emb], 1)

        w2 = self.W2(item_ids)
        b2 = self.b2(item_ids)

        # if for_pred:
        w2 = w2.squeeze()
        b2 = b2.squeeze()
        res = (x * w2).sum(1) + b2
        # else:
            #res = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze()
        out = torch.sigmoid(res)

        return out

    # def recommendation_score(self, session_ids, item_ids, item_history_ids): # for inference
    #     log_feats = self.log2feats(item_history_ids) # user_ids hasn't been used yet
    #     item_embs = self.item_emb(torch.LongTensor(item_ids)) # (U, I, C)

    #     final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

    #     logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

    #     return logits # preds # (U, I)

class SASRec(RecommenderModule):
    '''
    https://github.com/pmixer/SASRec.pytorch/blob/30c43cf090d429480339ab18d43354b3e399bc29/model.py#L5
    '''
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        path_item_embedding: str,
        from_index_mapping: str,
        freeze_embedding: bool,
        n_factors: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
        hist_size: int,
    ):
        super().__init__(project_config, index_mapping)
        self.path_item_embedding = path_item_embedding
        self.hist_size = hist_size
        self.n_factors = n_factors
        
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = load_embedding(self._n_items, n_factors, path_item_embedding, 
                                                from_index_mapping, index_mapping, freeze_embedding)

        self.pos_emb = torch.nn.Embedding(hist_size, n_factors) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=dropout)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(n_factors, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(n_factors, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(n_factors,
                                                            num_heads,
                                                            dropout)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(n_factors, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(n_factors, dropout)
            self.forward_layers.append(new_fwd_layer)


    def log2feats(self, log_seqs):
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5

        positions = np.tile(np.array(range(self.hist_size)), [log_seqs.shape[0], 1])
        
        seqs += self.pos_emb(torch.LongTensor(positions).to(log_seqs.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == 0)#.float()
        seqs *= (~timeline_mask).float().unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = (~torch.tril(torch.ones((tl, tl), dtype=torch.float)).bool()).float().to(log_seqs.device)

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  (~timeline_mask).float().unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, session_ids, item_ids, item_history_ids): # for training        
        log_feats = self.log2feats(item_history_ids) # (B, H, E)
        item_embs = self.item_emb(item_ids) # (B, E)
        #logits    = (log_feats * item_embs).sum(-1)#.mean(1)

        final_feat = log_feats[:, 0, :] # (B, E)  only use last QKV classifier, a waste

        #logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1) # (B, B)
        logits     = (final_feat * item_embs).sum(-1) # (B)

        return logits#.view(-1)

    # def recommendation_score(self, session_ids, item_ids, item_history_ids): # for inference
    #     log_feats = self.log2feats(item_history_ids) # user_ids hasn't been used yet
    #     item_embs = self.item_emb(torch.LongTensor(item_ids)) # (U, I, C)

    #     final_feat = log_feats[:, 0, :] # only use last QKV classifier, a waste

    #     logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

    #     return logits # preds # (U, I)

class GRURecModel(RecommenderModule):
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int,
        hidden_size: int,
        n_layers: int,
        path_item_embedding: str,
        from_index_mapping: str,
        dropout: float,
        freeze_embedding: bool
    ):
        super().__init__(project_config, index_mapping)
        self.path_item_embedding = path_item_embedding
        self.dropout = dropout
        self.hidden_size = hidden_size 
        self.n_layers = n_layers
        self.emb_dropout = nn.Dropout(0.25)

        self.item_embeddings = load_embedding(self._n_items, n_factors, path_item_embedding, 
                                                from_index_mapping, index_mapping, freeze_embedding)

        self.gru = nn.GRU(n_factors, self.hidden_size, self.n_layers, dropout=self.dropout)
        self.out = nn.Linear(self.hidden_size, self._n_items)
        self.sf  = nn.Softmax()

        self.weight_init = lecun_normal_init
        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)
            
    def flatten(self, input):
        return input.view(input.size(0), -1)

    def normalize(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if self.use_normalize:
            x = F.normalize(x, p=2, dim=dim)
        return x

    def forward(self, session_ids, item_ids, item_history_ids):
        embs = self.emb_dropout(self.item_embeddings(item_history_ids))
        
        output, hidden = self.gru(embs)
        #output = output.view(-1, output.size(2))  #(B,H)
        
        #out    = torch.softmax(self.out(output[:,-1]), dim=1)
        out    = self.out(output[:,-1])
        return out

    def recommendation_score(self, session_ids, item_ids, item_history_ids):
        
        scores = self.forward(session_ids, item_ids, item_history_ids)
        scores = scores[torch.arange(scores.size(0)),item_ids]

        return scores

class NARMModel(RecommenderModule):
    '''
    https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch.git
    '''
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int,
        n_layers: int,
        hidden_size: int,
        path_item_embedding: str,
        from_index_mapping: str,
        freeze_embedding: bool,        
        dropout: float        
    ):
        super().__init__(project_config, index_mapping)

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding_dim = n_factors
        #self.emb = nn.Embedding(self._n_items, self.embedding_dim, padding_idx = 0)

        self.emb = load_embedding(self._n_items, n_factors, path_item_embedding, 
                                                from_index_mapping, index_mapping, freeze_embedding)

        self.emb_dropout = nn.Dropout(0.25)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.n_layers)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(0.5)
        self.b = nn.Linear(self.embedding_dim, 2 * self.hidden_size, bias=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, session_ids, item_ids, item_history_ids):
        device = item_ids.device
        seq    = item_history_ids.permute(1,0)  #TODO

        hidden = self.init_hidden(seq.size(1)).to(device)
        embs = self.emb_dropout(self.emb(seq))
        #embs = pack_padded_sequence(embs, lengths)
        gru_out, hidden = self.gru(embs, hidden)
        #gru_out, lengths = pad_packed_sequence(gru_out)

        # fetch the last hidden state of last timestamp
        ht = hidden[-1]
        gru_out = gru_out.permute(1, 0, 2)

        c_global = ht
        q1 = self.a_1(gru_out.contiguous().view(-1, self.hidden_size)).view(gru_out.size())  
        q2 = self.a_2(ht)

        mask = torch.where(seq.permute(1, 0) > 0, torch.tensor([1.], device = device), torch.tensor([0.], device = device))

        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q2_masked = mask.unsqueeze(2).expand_as(q1) * q2_expand

        alpha = self.v_t(torch.sigmoid(q1 + q2_masked).view(-1, self.hidden_size)).view(mask.size())
        c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)

        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)
        
        item_embs = self.emb(torch.arange(self._n_items).to(device).long())
        scores = torch.matmul(c_t, self.b(item_embs).permute(1, 0))

        return scores

    def recommendation_score(self, session_ids, item_ids, item_history_ids):
        
        scores = self.forward(session_ids, item_ids, item_history_ids)
        scores = scores[torch.arange(scores.size(0)),item_ids]

        return scores

    def init_hidden(self, batch_size):
        return torch.zeros((self.n_layers, batch_size, self.hidden_size), requires_grad=True)
        
class MatrixFactorizationModel(RecommenderModule):
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int,
        path_item_embedding: str,
        from_index_mapping: str,
        dropout: float,
        freeze_embedding: bool,
        hist_size: int,
        weight_decay: float
    ):
        super().__init__(project_config, index_mapping)
        self.path_item_embedding = path_item_embedding
        self.hist_size = hist_size
        self.weight_decay = weight_decay # 0.025#1e-5

        self.hist_embeddings = nn.Embedding(self._n_items, n_factors)
        #self.hist_embeddings = load_embedding(self._n_items, n_factors, path_item_embedding, freeze_embedding)

        #self.hist_embeddings = load_embedding(self._n_items, n_factors, path_item_embedding, 
        #                                        from_index_mapping, index_mapping, freeze_embedding)



        self.item_embeddings = load_embedding(self._n_items, n_factors, path_item_embedding, 
                                                from_index_mapping, index_mapping, freeze_embedding)


        self.linear_w_emb = nn.Linear(n_factors*self.hist_size, n_factors)
        self.weight_emb =  nn.Parameter(torch.randn(self.hist_size, 1))
        self.use_normalize = True
        self.weight_init = lecun_normal_init

        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module):
        if type(module) == nn.Linear:
            self.weight_init(module.weight)
            module.bias.data.fill_(0.1)
            
    def flatten(self, input):
        return input.view(input.size(0), -1)

    def normalize(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if self.use_normalize:
            x = F.normalize(x, p=2, dim=dim)
        return x

    def forward(self, session_ids = None, item_ids = None, item_history_ids = None, negative_item_ids = None):
        # Item emb
        item_emb = self.normalize(self.item_embeddings(item_ids))

        # Item History embs
        hist_emb = self.normalize(self.hist_embeddings(item_history_ids), 2)

        #hist_emb_mean = torch.stack([self.linear_w_avg(emb) for emb in hist_emb], dim=0)
        #hist_mean_emb = hist_emb.mean(1)
        #hist_mean_emb = self.linear_w_emb(self.flatten(hist_emb))

        hist_mean_emb = torch.matmul(hist_emb.permute(0, 2, 1), self.weight_emb)
        hist_mean_emb = self.flatten(hist_mean_emb)

        dot_p = (item_emb * hist_mean_emb).sum(1)

        item_neg_emb = self.normalize(self.item_embeddings(negative_item_ids))

        dot_n = (item_neg_emb * hist_mean_emb).sum(1)

        diff  = dot_p - dot_n
        log_prob = F.logsigmoid(diff).sum()
        regularization = self.weight_decay * (hist_mean_emb.norm(dim=0).pow(2).sum() + item_emb.norm(dim=0).pow(2).sum())
        return -log_prob + regularization        

    def recommendation_score(self, session_ids, item_ids, item_history_ids):
        # Item emb
        item_emb = self.normalize(self.item_embeddings(item_ids))

        # Item History embs
        hist_emb = self.normalize(self.hist_embeddings(item_history_ids), 2)

        hist_mean_emb = torch.matmul(hist_emb.permute(0, 2, 1), self.weight_emb)
        hist_mean_emb = self.flatten(hist_mean_emb)

        dot_p = (item_emb * hist_mean_emb).sum(1)
        
        scores = torch.sigmoid(dot_p)
        
        return scores

class TripletNet(RecommenderModule):
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        n_factors: int, 
        use_normalize: bool,
        dropout: float,
        negative_random: float
    ):

        super().__init__(project_config, index_mapping)

        self.use_normalize   = use_normalize
        self.item_embeddings = nn.Embedding(self._n_items, n_factors)
        self.pos_embeddings = nn.Embedding(30, n_factors)

        self.negative_random = negative_random
        self.dropout = dropout
        #self.dropout_emb = EmbeddingDropout(dropout)
        self.dropout_emb = nn.Dropout(p=dropout)
        #self.weight_init = lecun_normal_init
        self.init_weights()
        #self.apply(self.init_weights)
        
    def embedded_dropout(self, embed, words, dropout=0.1):
        mask = embed.weight.data.new_empty((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
        return torch.nn.functional.embedding(words, masked_embed_weight)
                
    def init_weights(self):
        initrange = 0.1
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)

    def normalize(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if self.use_normalize:
            x = F.normalize(x, p=2, dim=dim)
        return x

    def get_harder_negative(self, item_ids: torch.Tensor, 
                            positive_item_ids: torch.Tensor, 
                            negative_item_list_idx: List[torch.Tensor]) -> Tuple[torch.Tensor]:

        batch_size = item_ids.size(0)

        anchors    = self.normalize(self.item_embeddings(item_ids.long()))  # (B, E)

        all_negative_items_embedding = self.normalize(
            self.item_embeddings(negative_item_list_idx), dim=2)  # (B, S, E)

        similarity_between_archor_and_negatives = (
            anchors.reshape(batch_size, 1, self.item_embeddings.embedding_dim) # (B, 1, E)
            * all_negative_items_embedding
        ).sum(
            2
        )  # (B, S)

        hardest_negative_items = torch.argmax(
            similarity_between_archor_and_negatives, dim=1
        )  # (B,)

        # negatives = all_negative_items_embedding[
        #     torch.arange(0, batch_size, device=item_ids.device), hardest_negative_items
        # ]

        negative_idx = negative_item_list_idx[
            torch.arange(0, batch_size, device=item_ids.device), hardest_negative_items
        ]
        return negative_idx

    def select_negative_item_emb(self, item_ids: torch.Tensor, positive_item_ids: torch.Tensor,
                        negative_list_idx: List[torch.Tensor] = None):

        if random.random() < self.negative_random:
            negative_item_idx = negative_list_idx[:,0]
        else:
            negative_item_idx = self.get_harder_negative(item_ids, positive_item_ids, negative_list_idx)
        
        return negative_item_idx

    def similarity(self, itemA, itemB):
        return torch.cosine_similarity(itemA, itemB)

    def forward(self, item_ids: torch.Tensor, 
                      positive_item_ids: torch.Tensor,
                      negative_list_idx: List[torch.Tensor] = None,
                      pos_relative = None) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        
        pos_relative_emb = self.normalize(self.pos_embeddings(pos_relative.long()))

        #arch_item_emb = self.embedded_dropout(self.item_embeddings, item_ids.long(), dropout=self.dropout if self.training else 0)
        arch_item_emb = self.normalize(self.item_embeddings(item_ids.long()))
        #arch_item_emb += pos_relative_emb

        #arch_item_emb = self.normalize(self.item_embeddings(item_ids.long()))
        #positive_item_emb = self.embedded_dropout(self.item_embeddings, positive_item_ids.long(), dropout=self.dropout if self.training else 0)
        positive_item_emb = self.normalize(self.item_embeddings(positive_item_ids.long()))
        positive_item_emb += pos_relative_emb
        #positive_item_emb = positive_item_emb - pos_relative_emb

        #raise(Exception(positive_item_emb.shape, pos_relative_emb.shape, arch_item_emb.shape))
        negative_item_ids = self.select_negative_item_emb(item_ids, positive_item_ids, negative_list_idx)
        #negative_item_emb = self.embedded_dropout(self.item_embeddings, negative_item_ids.long(), dropout=self.dropout if self.training else 0)
        negative_item_emb = self.normalize(self.item_embeddings(negative_item_ids.long()))
        
        #return self.dropout_emb(arch_item_emb, positive_item_emb, negative_item_emb)
        return self.dropout_emb(arch_item_emb), \
                self.dropout_emb(positive_item_emb), \
                self.dropout_emb(negative_item_emb)
        
        #return arch_item_emb, positive_item_emb, negative_item_emb, dot_arch_pos

class MLSASRec(RecommenderModule):
    '''
    https://github.com/pmixer/SASRec.pytorch/blob/30c43cf090d429480339ab18d43354b3e399bc29/model.py#L5
    '''
    def __init__(
        self,
        project_config: ProjectConfig,
        index_mapping: Dict[str, Dict[Any, int]],
        path_item_embedding: str,
        from_index_mapping: str,
        freeze_embedding: bool,
        n_factors: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
        hist_size: int,
    ):
        super().__init__(project_config, index_mapping)
        self.path_item_embedding = path_item_embedding
        self.hist_size = hist_size
        self.n_factors = n_factors
        
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = load_embedding(self._n_items, n_factors, path_item_embedding, 
                                                from_index_mapping, index_mapping, freeze_embedding)

        self.pos_emb = torch.nn.Embedding(hist_size, n_factors) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=dropout)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(n_factors, eps=1e-8)
        
        self.out = nn.Linear(self.n_factors, self._n_items)

        for _ in range(num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(n_factors, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(n_factors,
                                                            num_heads,
                                                            dropout)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(n_factors, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(n_factors, dropout)
            self.forward_layers.append(new_fwd_layer)


    def log2feats(self, log_seqs):
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5

        positions = np.tile(np.array(range(self.hist_size)), [log_seqs.shape[0], 1])
        
        seqs += self.pos_emb(torch.LongTensor(positions).to(log_seqs.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == 0)#.float()
        seqs *= (~timeline_mask).float().unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = (~torch.tril(torch.ones((tl, tl), dtype=torch.float)).bool()).float().to(log_seqs.device)

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  (~timeline_mask).float().unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        
        

        return log_feats

    def forward(self, session_ids, item_ids, item_history_ids): # for training        
        log_feats = self.log2feats(item_history_ids) # (B, H, E)
        item_embs = self.item_emb(item_ids) # (B, E)
        #logits    = (log_feats * item_embs).sum(-1)#.mean(1)

        final_feat = log_feats[:, 0, :] # (B, E)  only use last QKV classifier, a waste

        #logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1) # (B, B)
        #logits     = (final_feat * item_embs).sum(-1) # (B)

        output     = self.out(final_feat) #torch.softmax(self.out(final_feat), dim=1)
        #self.sf  = nn.Softmax()
        
        return output

    def recommendation_score(self, session_ids, item_ids, item_history_ids):
        
        scores = self.forward(session_ids, item_ids, item_history_ids)
        scores = scores[torch.arange(scores.size(0)),item_ids]

        return scores