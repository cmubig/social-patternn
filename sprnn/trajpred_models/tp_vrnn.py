# ------------------------------------------------------------------------------
# @file:    tp_vrnn.py
# @brief:   This class implements a simple VRNN-based trajectory prediction 
#           module.
#           Code based on: https://github.com/alexmonti19/dagnet
# ------------------------------------------------------------------------------
import json
from multiprocessing import context
import numpy as np
import torch
import torch.nn as nn

from torch import cat, tensor, zeros
from torch.autograd import Variable
from typing import Any, Tuple

from sprnn.trajpred_models.modeling.mlp import MLP
from sprnn.utils.common import dotdict

class VRNN(nn.Module):
    """ A class that implements trajectory prediction model using a VRNN """
    def __init__(self, config: dict, logger: Any, device: str = "cuda:0") -> None:
        """ Initializes the trajectory prediction network.
        
        Inputs:
        -------
        config[dict]: dictionary containing all configuration parameters.
        device[str]: device name used by the module. By default uses cuda:0. 
        """
        self._config = config
        super(VRNN, self).__init__()
        logger.debug("{} configuration:\n{}".format(
            self.name, json.dumps(self.config, indent=2)))

        self.device = device
        self.batch_size = self._config.batch_size
        logger.info(f"{self.name} uses torch.device({self.device})")
        
        self.dim = self.config.dim
        
        self.criterion = nn.MSELoss()
        
        # ----------------------------------------------------------------------
        # Model
        
        # x - feature extractor
        feat_enc = dotdict(self.config.feat_enc_x)
        self.f_x = MLP(feat_enc, logger, self.device)
        self.f_x_out_size = feat_enc.out_size
        
        # c - context extractor
        feat_ctx = dotdict(self.config.feat_enc_c)
        self.f_c = MLP(feat_ctx, logger, self.device)
        
        # x - encoder
        enc = dotdict(self.config.encoder)
        self.enc = MLP(enc, logger, self.device)
        self.enc_out_size = enc.out_size
        assert self.enc_out_size % 2 == 0, \
            f"Encoder's output size must be divisible by 2"
        self.z_dim = int(self.enc_out_size / 2)

        # x - prior
        self.prior = MLP(dotdict(self.config.prior), logger, self.device)

        # x - feature 
        self.f_z = MLP(dotdict(self.config.feat_enc_z), logger, self.device)
        
        # x - decoder
        self.dec = MLP(dotdict(self.config.decoder), logger, self.device)

        # recurrent network 
        rnn = dotdict(self.config.rnn)
        self.num_layers = rnn.num_layers
        self.rnn_dim = rnn.hidden_size
        self.rnn = nn.GRU(rnn.in_size, self.rnn_dim, self.num_layers)
            
        logger.info(f"{self.name} architecture:\n{self}")
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    @property
    def config(self) -> dict:
        return self._config

    def forward(self, hist: tensor, **kwargs) -> Tuple[tensor, tensor, Variable]:
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
        context = kwargs.get('context')

        for t in range(1, timesteps):          
            # extract location features
            x_t = hist[t]
            f_x_t = self.f_x(x_t) 
            
            # extract the context
            f_c_t = None if context is None else self.f_c(context[t])
            
            # cvae - encoder
            x_enc_embed = cat(
                [f_x_t, h[-1]], 1) if f_c_t is None else cat([f_x_t, f_c_t, h[-1]], 1)
            x_enc_t = self.enc(x_enc_embed)
            x_enc_mean_t = x_enc_t[:, :self.z_dim]
            x_enc_logvar_t = x_enc_t[:, self.z_dim:]

            # cvae - prior
            x_prior_embed = h[-1] if f_c_t is None else cat([f_c_t, h[-1]], 1)
            x_prior_t = self.prior(x_prior_embed)
            x_prior_mean_t = x_prior_t[:, :self.z_dim]
            x_prior_logvar_t = x_prior_t[:, self.z_dim:]

            # cvae - sample latent space
            z_t = self._reparameterize(x_enc_mean_t, x_enc_logvar_t)
            
            # cvae - extract features from latent space
            f_z_t = self.f_z(z_t)

            # cvae - decoder
            x_dec_embed = cat(
                [f_z_t, h[-1]], 1) if f_c_t is None else cat([f_z_t, f_c_t, h[-1]], 1)
            x_dec_t = self.dec(x_dec_embed)
            x_dec_mean_t = x_dec_t[:, :self.dim]
            x_dec_logvar_t = x_dec_t[:, self.dim:]

            # recurrence
            h_embedding = cat([f_x_t, f_z_t, f_c_t], 1).unsqueeze(0)
            _, h = self.rnn(h_embedding, h)

            # compute losses
            KLD += self._kld(
                x_enc_mean_t, x_enc_logvar_t, x_prior_mean_t, x_prior_logvar_t)
            NLL += self._nll_gauss(x_dec_mean_t, x_dec_logvar_t, x_t)

        return KLD, NLL

    @torch.no_grad()
    def evaluate(self, hist: tensor, **kwargs) -> Tuple[tensor, tensor, Variable]:
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
        context = kwargs.get('context')
        
        for t in range(1, timesteps):
            # extract location features
            x_t = hist[t]
            f_x_t = self.f_x(x_t) 
            
            # extract the context
            f_c_t = None if context is None else self.f_c(context[t])
            
            # cvae - encoder
            x_enc_embed = cat(
                [f_x_t, h[-1]], 1) if f_c_t is None else cat([f_x_t, f_c_t, h[-1]], 1)
            x_enc_t = self.enc(x_enc_embed)
            x_enc_mean_t = x_enc_t[:, :self.z_dim]
            x_enc_logvar_t = x_enc_t[:, self.z_dim:]

            # cvae - prior
            x_prior_embed = h[-1] if f_c_t is None else cat([f_c_t, h[-1]], 1)
            x_prior_t = self.prior(x_prior_embed)
            x_prior_mean_t = x_prior_t[:, :self.z_dim]
            x_prior_logvar_t = x_prior_t[:, self.z_dim:]

            # sample from latent space 
            z_t = self._reparameterize(x_enc_mean_t, x_enc_logvar_t)
            
            # extract features from latent 
            f_z_t = self.f_z(z_t)

            # cvae - decoder
            x_dec_embed = cat(
                [f_z_t, h[-1]], 1) if f_c_t is None else cat([f_z_t, f_c_t, h[-1]], 1)
            x_dec_t = self.dec(x_dec_embed)
            x_dec_mean_t = x_dec_t[:, :self.dim]
            x_dec_logvar_t = x_dec_t[:, self.dim:]

            # rnn
            h_embedding = cat([f_x_t, f_z_t, f_c_t], 1).unsqueeze(0)
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
        
        context = kwargs.get('context')
        samples = zeros(fut_len, batch_size, self.dim).to(self.device)
        
        for t in range(fut_len):
            # extract the context
            f_c_t = None if context is None else self.f_c(context[t])
            
            # cvae - prior
            x_prior_embed = h[-1] if f_c_t is None else cat([f_c_t, h[-1]], 1)
            x_prior_t = self.prior(x_prior_embed)
            x_prior_mean_t = x_prior_t[:, :self.z_dim]
            x_prior_logvar_t = x_prior_t[:, self.z_dim:]

            # sample from latent space 
            z_t = self._reparameterize(x_prior_mean_t, x_prior_logvar_t)
            
            # extract features from latent space
            f_z_t = self.f_z(z_t)

            # cvae - decoder
            x_dec_embed = cat(
                [f_z_t, h[-1]], 1) if f_c_t is None else cat([f_z_t, f_c_t, h[-1]], 1)
            x_dec_t = self.dec(x_dec_embed)
            x_dec_mean_t = x_dec_t[:, :self.dim]
            
            # (N, D)
            samples[t] = x_dec_mean_t.data

            # extract location features
            f_x_t = self.f_x(x_dec_mean_t)

            # rnn
            h_embedding = cat([f_x_t, f_z_t, f_c_t], 1).unsqueeze(0)
            _, h = self.rnn(h_embedding, h)

        return samples

    def _reparameterize(self, mean: tensor, log_var: tensor) -> tensor:
        """ Generates a sample z for the decoder using the mean, logvar parameters
        outputed by the encoder (during training) or prior (during inference). 
            z = mean + sigma * eps
        See: https://www.tensorflow.org/tutorials/generative/cvae
        
        Inputs:
        -------
        mean[tensor]: mean of a Gaussian distribution 
        log_var[tensor]: standard deviation of a Gaussian distribution.
                
        Outputs:
        --------
        z[tensor]: sampled latent value. 
        """
        logvar = torch.exp(log_var * 0.5).to(self.device)
        # eps is a random noise
        eps = torch.rand_like(logvar).to(self.device)
        return eps.mul(logvar).add(mean)

    def _kld(
        self, mean_enc: tensor, logvar_enc: tensor, mean_prior: tensor, 
        logvar_prior: tensor) -> tensor:
        """ KL Divergence between the encoder and prior distributions:
            x1 = log(sigma_p / sigma_e)
            x2 = sigma_m ** 2 / sigma_p ** 2
            x3 = (mean_p - mean_e) ** 2 / sigma_p ** 2
            KL(p, q) = 0.5 * (x1 + x2 + x3 - 1)
        See: https://stats.stackexchange.com/questions/7440/ \
                kl-divergence-between-two-univariate-gaussians
        
        Inputs:
        -------
        mean_enc[tensor]: encoder's mean at time t. 
        logvar_enc[tensor]: encoder's variance at time t.
        mean_prior[tensor]: prior's mean at time t. 
        logvar_prior[tensor]: prior's variance at time t.
        
        Outputs:
        --------
        kld[tensor]: Kullback-Leibler divergence between the prior and
        encoder's distributions time t. 
        """
        x1 = torch.sum((logvar_prior - logvar_enc), dim=1)
        x2 = torch.sum(torch.exp(logvar_enc - logvar_prior), dim=1)
        x3 = torch.sum((mean_enc - mean_prior).pow(2) /
                       (torch.exp(logvar_prior)), dim=1)
        kld_element = x1 - mean_enc.size(1) + x2 + x3
        return torch.mean(0.5 * kld_element)

    def _nll_gauss(self, mean: tensor, logvar: tensor, x: tensor) -> tensor:
        """ Negative Log-Likelihood with Gaussian.
            x1 = (x - mean) ** 2 / var
            x2 = logvar 
            x3 = const = 1 + log(2*pi)
            nll = 0.5 * (x1 + x2 + x3)
        See: https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
        
        Inputs:
        -------
        mean[tensor]: decoder's mean at time t.
        logvar[tensor]: decoder's variance a time t.
        x[tensor]: ground truth X at time t.
        
        Outpus:
        -------
        nll[tensor]: Gaussian Negative Log-Likelihood at time t. 
        """
        x1 = torch.sum(((x - mean).pow(2)) / torch.exp(logvar), dim=1)
        x2 = x.size(1) * np.log(2 * np.pi)
        x3 = torch.sum(logvar, dim=1)
        nll = torch.mean(0.5 * (x1 + x2 + x3))
        return nll
    
    def _ce(self, pred_x: tensor, gt_x: tensor) -> tensor:
        """ Cross-Entropy loss between ground truth tensor (gt_x) and predicted
        tensor (pred_x). 
        
        Inputs:
        -------
        pred_x[tensor]: predicted motion primitive class
        gt_x[tensor]: ground truth motion primitive class
        
        Outpus:
        --------
        ce[torch.Float]: cross-entropy loss value
        """
        return -torch.sum(gt_x * pred_x)
    
    def _mse(self, pred_x: tensor, gt_x: tensor) -> tensor:
        """ Mean Squared Error between ground truth tensor (gt_x) and predicted
        tensor (pred_x). 
        
        Inputs:
        -------
        pred_x[tensor]: predicted patterns
        gt_x[tensor]: ground truth patterns
        
        Outpus:
        --------
        mse[torch.Float]: mean squared error value
        """
        return torch.sqrt(self.criterion(self.scale * pred_x, self.scale * gt_x))