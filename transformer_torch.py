
import torch
from torch import nn
import torch.nn.functional as F
import math 

def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false
    In place operation
    :param tns:
    :return:
    """

    b, h, w = matrices.size()

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval
    
def contains_nan(tensor):
    return bool((tensor != tensor).sum() > 0)


class SelfAttention(nn.Module):
  def __init__(self, k, heads=8):
    super().__init__()
    self.k, self.heads = k, heads
    
    # These compute the queries, keys and values for all
    # heads (as a single concatenated vector)
    self.tokeys    = nn.Linear(k, k * heads, bias=False)
    self.toqueries = nn.Linear(k, k * heads, bias=False)
    self.tovalues  = nn.Linear(k, k * heads, bias=False)

 	# This unifies the outputs of the different heads into
 	# a single k-vector
    self.unifyheads = nn.Linear(heads * k, k)
    
  def forward(self, x):
      b, t, k = x.size()
      
      h = self.heads
      
      queries = self.toqueries(x).view(b,t,h,k)# (b, t,h*k) --> (b,y,h,k) each head with its own vector 
      keys = self.tokeys(x).view(b,t,h,k) 
      values = self.tovalues(x).view(b,t,h,k) 
      
      keys = keys.transpose(1, 2).contiguous().view(b*h, t, k) 
      queries = queries.transpose(1, 2).contiguous().view(b*h, t, k) 
      values = values.transpose(1, 2).contiguous().view(b*h,t,k)
      
      queries = queries / (k**(1/4)) 
      keys = keys / (k**(1/4)) 
  
      dot = torch.bmm(queries, keys.transpose(1,2)) 
      dot = F.softmax(dot, dim=2) 
      out = torch.bmm(dot, values).view(b, h, t, k) 
      out = out.transpose(1, 2).contiguous().view(b, t, h*k) 
      return self.unifyheads(out) 
  
class SelfAttention2(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        """
        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        keys    = self.tokeys(x)   .view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = dot / math.sqrt(e) # dot contains b*h  t-by-t matrices with raw self-attention logits

        assert dot.size() == (b*h, t, t), f'Matrix has size {dot.size()}, expected {(b*h, t, t)}.'

        if self.mask: # mask out the lower half of the dot matrix,including the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2) # dot now has row-wise self-attention probabilities

        assert not contains_nan(dot[:, 1:, :]) # only the forst row may contain nan

        if self.mask == 'first':
            dot = dot.clone()
            dot[:, :1, :] = 0.0
            # - The first row of the first attention matrix is entirely masked out, so the softmax operation results
            #   in a division by zero. We set this row to zero by hand to get rid of the NaNs

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)   
class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x
    
    
layer = SelfAttention(10, heads = 8)
x = torch.rand(size = (1,3,10))
y = layer.forward(x)

