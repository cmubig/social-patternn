
# ------------------------------------------------------------------------------
# @file:    mha.py
# @brief:   This class implements a simple Multi-Head Attention module. 
# ------------------------------------------------------------------------------
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any

class MHA(nn.Module):
    def __init__(self, config: dict, logger: Any, device: str = "cuda:0") -> None:
        """ Implements a simple Multi-Head Attention. 
        
        Inputs
        ------
        config[dict]: dictionary containing all network configuration parameters.
        device[str]: device used by the module. 
        """
        self._name = self.__class__.__name__
        super(MHA, self).__init__()
        
        self._config = config
        logger.info("{} configuration:\n{}".format(
            self.name, json.dumps(self.config, indent=2)))
        
        self.device = device
        logger.info(f"{self.name} uses torch.device({self.device})")
        
        self.dropout = self.config.dropout
        self.in_size = self.config.in_size 
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_heads
        self.head_size = self.hidden_size // self.num_heads
        
        # architecture 
        self.qkv_proj = nn.Linear(self.in_size, self.hidden_size * 3)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self._reset_parameters()
        
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def config(self)-> dict:
        return self._config
    
    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        
    def _scaled_dot_product(self, q, k, v, mask=None):
        d_k = q.size()[-1]
    
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
            
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

    def forward(
        self, x: torch.tensor, return_attention: bool = False, mask = None
    ) -> torch.tensor:
        """ Forward propagation of x.
        
        Inputs
        ------
        x[torch.tensor]: tensor of shape (batch size, in size)
        return_attention[bool]: if True, returns attention mask
     
        Outputs
        -------
        x[torch.tensor]: encoded tensor of shape (batch size, out size)
        """ 
        batch_size, seq_len, hidden_dim = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3*self.head_size)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1) # [Batch, Head, SeqLen, Dims / 3]
        
        # Determine value outputs
        values, attention = self._scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_len, self.hidden_size)
        x = self.o_proj(values)
        
        if return_attention:
            return x, attention    
        return x