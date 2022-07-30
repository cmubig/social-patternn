# ------------------------------------------------------------------------------
# @file:    tp_vrnn.py
# @brief:   This class implements a simple VRNN-based trajectory prediction 
#           module.
#           Code based on: https://github.com/alexmonti19/dagnet
# ------------------------------------------------------------------------------
import json
import numpy as np
import torch
import torch.nn as nn

from torch import cat, tensor, zeros
from torch.autograd import Variable
from typing import Any, Tuple

from sprnn.trajpred_models.tp_vrnn import VRNN
from sprnn.trajpred_models.modeling.mlp import MLP
from sprnn.utils.common import dotdict

class SocialVRNN(VRNN):
    """ A class that implements trajectory prediction model using a VRNN """
    def __init__(self, config: dict, logger: Any, device: str = "cuda:0") -> None:
        """ Initializes the trajectory prediction network.
        
        Inputs:
        -------
        config[dict]: dictionary containing all configuration parameters.
        device[str]: device name used by the module. By default uses cuda:0. 
        """
        # Intitializes base the base model which is a VRNN module
        super().__init__(config, logger, device)
        
        logger.info(f"{self.name} architecture:\n{self}")
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    @property
    def config(self) -> dict:
        return self._config

    def forward(self, hist: tensor) -> Tuple[tensor, tensor]:
        """ Forward propagation of observed trajectories.
        
        Inputs:
        -------
        hist[tensor(hist_len, batch_size, dims)]: trajectory histories 
        kwargs: keyword-based arguments
            
        Outputs:
        --------        import pdb; pdb.set_trace()
        KLD[tensor]: accumulated KL divergence values
        NLL[tensor]: accumulated Neg Log-Likelyhood values
        h[tensor(num_rnn_layers, batch_size, r)]: tensor
        """
        timesteps, batch_size, _ = hist.shape
      
        KLD = zeros(1).to(self.device)
        NLL = zeros(1).to(self.device)
        h = Variable(zeros(
            self.num_layers, batch_size, self.rnn_dim)).to(self.device)
        
        for t in range(1, timesteps):
            # x - extract features at step t
            x_t = hist[t]
            f_x_t = self.f_x(x_t) 
            
            # x - encode step t (encoder)
            x_enc_embedding = cat([f_x_t, h[-1]], 1)
            x_enc_t = self.enc(x_enc_embedding)
            x_enc_mean_t = x_enc_t[:, :self.z_dim]
            x_enc_logvar_t = x_enc_t[:, self.z_dim:]

            # x - encode step t (prior)
            x_prior_t = self.prior(h[-1])
            x_prior_mean_t = x_prior_t[:, :self.z_dim]
            x_prior_logvar_t = x_prior_t[:, self.z_dim:]

            # z - sample from latent space 
            z_t = self._reparameterize(x_enc_mean_t, x_enc_logvar_t)
            
            # z - extract feature at step t
            f_z_t = self.f_z(z_t)

            # z - decode step t to generate x_t
            x_dec_embedding = cat([f_z_t, h[-1]], 1)
            x_dec_t = self.dec(x_dec_embedding)
            x_dec_mean_t = x_dec_t[:, :self.dim]
            x_dec_logvar_t = x_dec_t[:, self.dim:]

            # recurrence
            h_embedding = cat([f_x_t, f_z_t], 1).unsqueeze(0)
            _, h = self.rnn(h_embedding, h)

            # compute losses
            KLD += self._kld(
                x_enc_mean_t, x_enc_logvar_t, x_prior_mean_t, x_prior_logvar_t)
            NLL += self._nll_gauss(x_dec_mean_t, x_dec_logvar_t, x_t)

        return KLD, NLL

    @torch.no_grad()
    def evaluate(self, hist: tensor) -> Tuple[tensor, tensor, Variable]:
        """ Forward propagation of observed trajectories.
        
        Inputs:
        -------
        hist[tensor(hist_len, batch_size, dims)]: trajectory histories 
        kwargs: keyword-based arguments
            
        Outputs:
        --------
        KLD[tensor]: accumulated KL divergence values
        NLL[tensor]: accumulated Neg Log-Likelyhood values
        h[tensor(num_rnn_layers, batch_size, r)]: tensor
        """
        timesteps, batch_size, _ = hist.shape
      
        KLD = zeros(1).to(self.device)
        NLL = zeros(1).to(self.device)
        h = Variable(zeros(
            self.num_layers, batch_size, self.rnn_dim)).to(self.device)
        
        for t in range(1, timesteps):
            # x - extract features at step t
            x_t = hist[t]
            f_x_t = self.f_x(x_t) 
            
            # x - encode step t (encoder)
            x_enc_embedding = cat([f_x_t, h[-1]], 1)
            x_enc_t = self.enc(x_enc_embedding)
            x_enc_mean_t = x_enc_t[:, :self.z_dim]
            x_enc_logvar_t = x_enc_t[:, self.z_dim:]

            # x - encode step t (prior)
            x_prior_t = self.prior(h[-1])
            x_prior_mean_t = x_prior_t[:, :self.z_dim]
            x_prior_logvar_t = x_prior_t[:, self.z_dim:]

            # z - sample from latent space 
            z_t = self._reparameterize(x_enc_mean_t, x_enc_logvar_t)
            
            # z - extract feature at step t
            f_z_t = self.f_z(z_t)

            # z - decode step t to generate x_t
            x_dec_embedding = cat([f_z_t, h[-1]], 1)
            x_dec_t = self.dec(x_dec_embedding)
            x_dec_mean_t = x_dec_t[:, :self.dim]
            x_dec_logvar_t = x_dec_t[:, self.dim:]

            # recurrence
            h_embedding = cat([f_x_t, f_z_t], 1).unsqueeze(0)
            _, h = self.rnn(h_embedding, h)

            # compute losses
            KLD += self._kld(
                x_enc_mean_t, x_enc_logvar_t, x_prior_mean_t, x_prior_logvar_t)
            NLL += self._nll_gauss(x_dec_mean_t, x_dec_logvar_t, x_t)

        return KLD, NLL, h
    
    @torch.no_grad()
    def inference(self, fut_len: int, h: Variable, **kwargs) -> tensor:
        """ Inference (sampling) trajectories.
        
        Inputs:
        -------
        fut_len[int]: length of the predicted trajectory
        h[torch.Variable(rnn_layers, batch_size, dim)]: torch.Variable 
        kwargs: any other keyword-based arguments
        
        Outputs:
        --------
        sample[tensor(fut_len, batch_size, dims)]: predicted trajectories
        """
        _, batch_size, _ = h.shape

        samples = zeros(fut_len, batch_size, self.dim).to(self.device)
        
        for t in range(fut_len):
            # x - encode hidden state to generate latent space (prior)
            x_prior_t = self.prior(h[-1])
            x_prior_mean_t = x_prior_t[:, :self.z_dim]
            x_prior_logvar_t = x_prior_t[:, self.z_dim:]

            # z - sample from latent space 
            z_t = self._reparameterize(x_prior_mean_t, x_prior_logvar_t)
            
            # z - extract feature at step t
            f_z_t = self.f_z(z_t)

            # z - decode step t to generate x_t
            x_dec_embedding = cat([f_z_t, h[-1]], 1)
            x_dec_t = self.dec(x_dec_embedding)
            x_dec_mean_t = x_dec_t[:, :self.dim]
            
            # (N, D)
            samples[t] = x_dec_mean_t.data

            # x - extract features from decoded latent space (~ 'x')
            f_x_t = self.f_x(x_dec_mean_t)

            # recurrence
            h_embedding = cat([f_x_t, f_z_t], 1).unsqueeze(0)
            _, h = self.rnn(h_embedding, h)

        return samples