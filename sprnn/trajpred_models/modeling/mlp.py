
# ------------------------------------------------------------------------------
# @file:    mlp.py
# @brief:   This file contains the implementation of a Multi-Layer Perceptron 
#           (MLP) network. 
# ------------------------------------------------------------------------------
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any

class MLP(nn.Module):
    def __init__(self, config: dict, logger: Any, device: str = "cuda:0") -> None:
        """ Implments a simple MLP with ReLU activations. 
        
        Inputs:
        -------
        config[dict]: network configuration parameters.
        logger[logging]: logger where ouputs are being written.
        device[str]: device used by the module. 
        """
        self._name = self.__class__.__name__
        super(MLP, self).__init__()
        
        self._config = config
        logger.debug("{} configuration:\n{}".format(
            self.name, json.dumps(self.config, indent=2)))
        
        self.device = device
        logger.debug(f"{self.name} uses torch.device({self.device})")
        
        self.dropout = self.config.dropout
        
        self.layer_norm = False
        if config.layer_norm:
            self.layer_norm = config.layer_norm
        
        self.relu = True
        if config.relu:
            self.relu = config.relu 
            
        # Network architecture 
        feats = [config.in_size, *config.hidden_size, config.out_size]
        mlp = []
        for i in range(len(feats)-1):
            mlp.append(nn.Linear(in_features=feats[i], out_features=feats[i+1]))
            if self.layer_norm:
                mlp.append(nn.LayerNorm(normalized_shape=feats[i+1]))
        
        if config.softmax:
            mlp.append(nn.Softmax(dim=-1))
            
        self.net = nn.ModuleList(mlp)
        
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def config(self)-> dict:
        return self._config
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        """ Forward propagation of x.
        
        Inputs:
        -------
        x[torch.tensor(batch_size, input_size)]: input tensor
            
        Outputs:
        -------
        x[torch.tensor(batch_size, output_size)]: output tensor
        """ 
        for i in range(len(self.net)-1):
            x = self.net[i](x)
            if self.relu:
                x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout)
        x = self.net[-1](x)
        return x