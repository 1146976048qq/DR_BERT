import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import logging

class Embeddings(nn.Module):
    """
    This is word embeddings
    """
    def __init__(self, vocab_size, d_model):
        super(Embeddings, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model # word_vec_size, 512
        self.build_model()
        
    def build_model(self):
        self.word_embeddings = nn.Embedding(self.vocab_size, self.d_model)
        
    def forward(self, x):
        return self.word_embeddings(x)*math.sqrt(self.d_model)
    
    
class PositionEncoding(nn.Module):
    """
    This is the special position encoding used in transformer
    """
    def __init__(self, d_model, keep_prob, max_len=5000):
        super(PositionEncoding,self).__init__()
        self.d_model = d_model # word_vec_size, 512
        self.keep_prob = keep_prob
        self.max_len = max_len
        self.build_model()
        
    def build_model(self):
        self.dropout = nn.Dropout(self.keep_prob)
        # compute position encoding
        self.pe = torch.zeros(self.max_len, self.d_model)
        #position = torch.arange(0,self.max_len).unsqueeze(1)
        #div_term = torch.exp(torch.arange(0,self.d_model,2)*(-math.log(10000.0)/self.d_model))
        position = torch.arange(0., self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., self.d_model, 2) * -(math.log(10000.0) / self.d_model))
        self.pe[:,0::2] = torch.sin(position*div_term)
        self.pe[:,1::2] = torch.cos(position*div_term)
        self.pe = self.pe.unsqueeze(0)
    
    def forward(self, x):
        x = x + Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dropout(x)
    
    
class PositionwiseFeedForward(nn.Module):
    """
    This is the positionwise Feed Forward layer
    """
    def __init__(self, d_model, d_ff, keep_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.keep_prob = keep_prob
        self.build_model()
        
    def build_model(self):
        self.W_1 = nn.Linear(self.d_model, self.d_ff)
        self.W_2 = nn.Linear(self.d_ff, self.d_model)
        self.dropout = nn.Dropout(self.keep_prob)
        
    def forward(self, x):
        """
        should be max(0,xW_1+b1)W_2+b_2
        """
        return self.W_2(self.dropout(F.relu(self.W_1(x))))
    
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, keep_prob=0.1):
        super(ScaledDotProductAttention,self).__init__()
        self.keep_prob = keep_prob
        self.build_model()
        
    def build_model(self):
        self.dropout = nn.Dropout(self.keep_prob)
        
    def forward(self,query,key,value,mask=None):
        """
        Args:
            query: (batch, heads, q_len, d_k)
            key: (batch, heads,k_len, d_k)
            value: (batch, heads,k_len, d_k)
        """
        d_k = query.size(3)
        seq_len = query.size(2)
        heads = query.size(1)
        scores = torch.matmul(query, key.transpose(-2,-1))/math.sqrt(d_k)
        # scores (batch,heads, q_len, k_len)
        if mask is not None:
            scores = scores.masked_fill(mask==0,-1e9)
        p_attn = F.softmax(scores,dim=-1)
        # p_attn: (batch, heads, q_len, k_len)
        logging.debug('scaleddot attention: p_attn shape {}'.format(p_attn.shape))
        p_attn = self.dropout(p_attn)
        p_attn = torch.matmul(p_attn, value)
        # p_attn shape: (batch, heads, q_len, d_k)
        p_attn = p_attn.transpose(2,1).contiguous().view(-1, seq_len, heads*d_k)
        #return torch.matmul(p_attn,value), p_attn
        return p_attn

def clones(module, N):
    """
    Produce N identical layers
    """
    from copy import deepcopy
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    """
    multihead attention, h is number of heads
    """
    def __init__(self, hidden_size, heads, keep_prob=0.1):
        super(MultiHeadAttention,self).__init__()
        assert hidden_size % heads == 0
        self.hidden_size = hidden_size
        self.heads = heads
        self.keep_prob = keep_prob
        self.d_k = self.hidden_size // self.heads
        self.build_model()
        
    def build_model(self):
        self.dropout = nn.Dropout(self.keep_prob)
        self.linears = clones(nn.Linear(self.hidden_size, self.hidden_size), 4)
        self.scaleddotattn = ScaledDotProductAttention(self.keep_prob)
        self.attn = None
        
    def forward(self, inputs, mask=None):
        """
        Args:
            inputs: (batch, seq_len, d_model)
        """
        if mask is not None:
            # same mask applied to all h heads
            mask = mask.unsqueeze(1)
        seq_len = inputs.size(1)

        # first, do all the linear projections in batch from d_model to h*d_k
        query, key, value = [l(inputs).view(-1,seq_len,self.heads,self.d_k).transpose(1,2) for l in self.linears[:3]]
        #query,key,value = [l(x).view(-1,seq_len,self.h,self.d_k).transpose(1,2) for l,x in zip(self.linears,(inputs,inputs,inputs))]
        logging.debug('multihead attnention: query shape {}, key shape {}, value shape {}'.format(query.shape, key.shape, value.shape))
        
        # second, apply attention on all the projected vectors in batch
        #x,self.attn = self.scaleddotattn(query,key,value,mask)
        x = self.scaleddotattn(query, key, value, mask)
        logging.debug('multihead attention: x {}'.format(x.shape))
        
        # third, concat using a view and apply a final linear
        x = self.linears[-1](x)
        #x = x.transpose(1,2).contiguous().view(-1,seq_len, self.h*self.d_k)
        logging.debug('multihead attention: x {}'.format(x.shape))
        return x
        #return self.linears[-1](x)
    

class LayerNorm(nn.Module):
    """
    There is implementation in nn.LayerNorm
    """
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.features = features
        self.eps = eps
        self.build_model()
        
    def build_model(self):
        self.a_2 = nn.Parameter(torch.ones(self.features))
        self.b_2 = nn.Parameter(torch.zeros(self.features))
        
    def forward(self, x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        logging.debug('LayerNorm: mean shape {}, std shape {}'.format(mean.shape, std.shape))
        return self.a_2*(x-mean)/(std + self.eps)+self.b_2
    

class SublayerConnection(nn.Module):
    def __init__(self,size,keep_prob):
        super(SublayerConnection,self).__init__()
        self.keep_prob = keep_prob
        self.size = size
        self.build_model()
        
    def build_model(self):
        self.dropout = nn.Dropout(self.keep_prob)
        self.norm = LayerNorm(self.size)
        
    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size
        """
        return x+self.dropout(sublayer(self.norm(x)))
    
    
class EncoderLayer(nn.Module):
    """
    This is just one layer of encoder,contains one multihead attention and one positionwise feed forward layer
    """
    def __init__(self,size,self_attn,feed_forward,keep_prob):
        super(EncoderLayer,self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.keep_prob = keep_prob
        self.build_model()
    
    def build_model(self):
        self.sublayer = clones(SublayerConnection(self.size,self.keep_prob),2)
        
    def forward(self, x, mask):
        """
        one encoder layer
        """
        x = self.sublayer[0](x, lambda x:self.self_attn(x,x,x,mask))
        return self.sublayer[1](x, self.feed_forward)
    
    
class Encoder(nn.Module):
    """
    N layers of encoder
    """
    def __init__(self, layer, N=6):
        super(Encoder, self).__init__()
        self.layer = layer # encoder layer
        self.N = N
        self.build_model()
        
    def build_model(self):
        self.layers = clones(self.layer, self.N)
        self.norm = LayerNorm(self.layer.size)
    
    def forward(self, x, mask):
        """
        Pass the input and mask through each layer
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
    
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, keep_prob):
        super(DecoderLayer,self).__init__()
        self.size = size 
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.src_attn = src_attn
        self.keep_prob = keep_prob
        self.build_model()
        
    def build_model(self):
        self.sublayer = clones(SublayerConnection(self.size,self.keep_prob), 3)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        """
        one decoder layer
        Args:
            x: target
            memory: output from encoder
            src_mask: mask for source input
            tgt_mask: mask for target input
        """
        m = memory
        x = self.sublayer[0](x, lambda x:self.self_attn(x,x,x,tgt_mask))
        x = self.sublayer[1](x, lambda x:self.src_attn(x,m,m,src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
    
class Decoder(nn.Module):
    """
    N layers of decoder 
    """
    def __init__(self, layer, N=6):
        super(Decoder, self).__init__()
        self.layer = layer # decoder layer
        self.N = N
        self.build_model()
        
    def build_model(self):
        self.layers = clones(self.layer, self.N)
        self.norm = LayerNorm(self.layer.size)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Args:
            x: target
            memory: output from encoder
            src_mask: mask for source input
            tgt_mask: mask for target input
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
    
class Generator(nn.Module):
    """
    Final Layer after decoder layers, include linear and softmax, returns probability
    """
    def __init__(self,vocab_size, d_model):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.build_model()
        
    def build_model(self):
        self.proj = nn.Linear(self.d_model, self.vocab_size)
        
    def forward(self, x):
        return F.log_softmax(self.proj(x),dim=-1)
    

class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,src_embed,tgt_embed,generator):
        super(EncoderDecoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def encode(self, src, src_mask):
        """
        Args:
            src: source sentence (batch, seq_len)
            src_mask: mask for source sentence
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        Args:
            memory: output from encoder
            src_mask: mask for source input
            tgt: target sentence
            tgt_mask: mask for target input
        """
        return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)
    
    def forward(self,src,tgt,src_mask,tgt_mask):
        """
        process source and target input
        """
        return self.decode(self.encode(src,src_mask),src_mask,tgt,tgt_mask)
    
    
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_model(src_vocab,tgt_vocab,N=6,d_model=512,d_ff=2048,h=8,keep_prob=0.1):
    """
    Helper: construct transformer encoder-decoder
    """
    from copy import deepcopy
    attn = MultiHeadAttention(d_model,h,keep_prob)
    ff = PositionwiseFeedForward(d_model,d_ff,keep_prob)
    position = PositionEncoding(d_model,keep_prob)
    model = EncoderDecoder(Encoder(EncoderLayer(d_model,deepcopy(attn),deepcopy(ff),keep_prob),N),
                          Decoder(DecoderLayer(d_model,deepcopy(attn),deepcopy(attn),deepcopy(ff),keep_prob),N),
                          nn.Sequential(Embeddings(src_vocab,d_model),deepcopy(position)),
                           nn.Sequential(Embeddings(tgt_vocab,d_model),deepcopy(position)),
                           Generator(tgt_vocab,d_model)
                          )
    # initialize parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model