import time
import torch
from torch import  nn , Tensor
import math 
import onnxruntime as ort

import warnings
warnings.filterwarnings("ignore")

from torch.nn import TransformerEncoderLayer


class CausalTransformerEncoderLayer(TransformerEncoderLayer):
    """Transformer encoder layer with causal mask.

    See :class:`torch.nn.TransformerEncoderLayer` for details.

    Examples:

        >>> L, N, E = 5, 1, 2  # sequence length, batch, features
        >>> m = CausalTransformerEncoderLayer(E, 1)
        >>> src = torch.empty(L, N, E)
        >>> m.causal_mask(src)
        tensor([[False,  True,  True,  True,  True],
                [False, False,  True,  True,  True],
                [False, False, False,  True,  True],
                [False, False, False, False,  True],
                [False, False, False, False, False]])
        >>> assert m(src).size() == src.size()
    """

    def causal_mask(self, src: Tensor) -> Tensor:
        # In PyTorch documentation of MultiHeadAttention:
        # > (L, S) where L is the target sequence length,
        # > S is the source sequence length.
        query, key, value = src, src, src
        trues = torch.ones(
            (query.size(0), key.size(0)), dtype=torch.bool, device=src.device
        )
        return trues.triu(diagonal=1)

    def forward(self, src: Tensor, *args, **kwargs) -> Tensor:
        return super().forward(src, src_mask=self.causal_mask(src))

    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]

        Returns:
            output Tensor of shape [batch_size, seq_len, d_model]
        """

        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)
    
class Embedder(nn.Module):
    def __init__(self,in_channels,d_model=64,kernel_size=3,stride=1,padding='same') -> None:
        super().__init__()
        # self.conv = torch.nn.Conv1d(in_channels,d_model,kernel_size=kernel_size,stride=stride,padding=padding)
        self.conv = nn.Linear(in_channels,d_model,bias=False)
    
    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, d_model, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len, d_model]
        """
        return self.conv(src)
    

class InferenceEncoder(nn.Module):
    def __init__(self,in_chans=50,
        seq_len = 100,
        embed_dim=64,
        depth=2,
        num_heads=4,
        dropout = 0.1,
        norm_first = True,
        norm_layer=nn.LayerNorm,
        d_hid = 128,) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.embedder = Embedder(in_chans,embed_dim)
        self.pos_encoder_e = PositionalEncoding(embed_dim,dropout,max_len=seq_len)

        self.blocks = nn.ModuleList(
            [
            
                CausalTransformerEncoderLayer(embed_dim, num_heads, d_hid, dropout,
                                            norm_first=norm_first)
                
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, d_model, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len, d_model]
        """
        x = self.embedder(x)
        x = self.pos_encoder_e(x * math.sqrt(self.embed_dim))
        x = x.permute(1,0,2)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x

size1_args = {'in_chans' : 75,
              'seq_len' : 100,
              'embed_dim' : 64,
              'depth' : 4,
              'num_heads' : 4,
              'd_hid' : 128
              }
size2_args = {'in_chans' : 75,
              'seq_len' : 100,
              'embed_dim' : 128,
              'depth' : 6,
              'num_heads' : 4,
              'd_hid' : 128
              }
size3_args = {'in_chans' : 75,
              'seq_len' : 100,
              'embed_dim' : 64,
              'depth' : 2,
              'num_heads' : 4,
              'd_hid' : 128
              }
with torch.no_grad():
    X10 =  torch.rand(1,10,75)
    X50 =  torch.rand(1,50,75)
    X100 =  torch.rand(1,100,75)
    model = InferenceEncoder(**size1_args)
    
    model.eval()
    pytorch_total_params = sum(p.numel() for p in model.parameters())

    print(f'num params : { pytorch_total_params / 1e6} millions ' )
    model_jit = torch.jit.trace(model,X10)
    model_jit = torch.jit.freeze(model_jit)
    fs = time.time()
    model_jit(X10)
    print(f'first inference takes {(time.time() - fs)*1000} ms')

    print(f'seq len 10')
    start = time.time()

    for i in range(1000):
        z = model_jit(X10)

    end = time.time()
    inf_time = end-start
    print(f'total inference time  {inf_time}')
    print(f'inference time per sample  {inf_time} ms')
    
    print(f'seq len 50')
    model_jit = torch.jit.trace(model,X50)
    model_jit(X50)
    start = time.time()

    for i in range(1000):
        z = model_jit(X50)

    end = time.time()
    inf_time = end-start
    print(f'total inference time  {inf_time}')
    print(f'inference time per sample  {inf_time} ms')
    
    print(f'seq len 100')
    model_jit = torch.jit.trace(model,X100)
    model_jit(X100)
    start = time.time()

    for i in range(1000):
        z = model_jit(X100)

    end = time.time()
    inf_time = end-start
    print(f'total inference time  {inf_time}')
    print(f'inference time per sample  {inf_time} ms')
from torch.profiler import profile, record_function, ProfilerActivity

    
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(X100)
        
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

   