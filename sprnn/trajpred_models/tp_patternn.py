# ------------------------------------------------------------------------------
# @file:    tp_patternn.py
# @brief:   This class extends the TrajPredVRNN class with a pattern learning
#           module.
#           Code based on: sprnn.trajpred_models.tp_vrnn.TrajPredVRNN
# ------------------------------------------------------------------------------
import torch

from torch import tensor, cat, zeros
from torch.autograd import Variable
from typing import Any, Tuple

from sprnn.trajpred_models.modeling.mlp import MLP
from sprnn.trajpred_models.tp_vrnn import VRNN
from sprnn.utils.common import dotdict

class PatteRNN(VRNN):
    """ A class that implements trajectory prediction model using a VRNN """
    def __init__(self, config: dict, logger: Any, device: str = "cpu") -> None:
        """ Initializes the trajectory prediction network.
        
        Inputs:
        -------
        config[dict]: dictionary containing all configuration parameters.
        device[str]: device name used by the module. By default uses cpu. 
        """
        # Intitializes base the base model consisting of a VRNN module
        super().__init__(config, logger, device)

        # PatternNet:
        self.scale = config.scale
        
        pattern_net = dotdict(self.config.pattern_net)
        self.pat_len = pattern_net.pat_len
        
        # pattern_net - feature extractor
        feat_pat = dotdict(pattern_net.feat_pat)
        feat_pat.in_size = self.pat_len * self.dim 
        self.f_pat = MLP(feat_pat, logger, self.device)
        
        # pattern_net - pattern decoder 
        dec_pat = dotdict(pattern_net.dec_pat)
        dec_pat.out_size = self.pat_len * self.dim
        self.dec_pat = MLP(dec_pat, logger, self.device)
        
        logger.info(f"{self.name} architecture:\n{self}")

    def forward(
        self, hist: tensor, pat: tensor, **kwargs
    ) -> Tuple[tensor, tensor, tensor]:
        """ Forward propagation of observed trajectories.
        
        Inputs:
        -------
        hist[tensor(hist_len, batch_size, dims)]: trajectory histories 
        kwargs: keyword-based arguments
            
        Outputs:
        --------
        KLD[tensor]: accumulated KL divergence values
        NLL[tensor]: accumulated Neg Log-Likelyhood values
        MSE[tensor]: accumulated Mean Squared Error values
        h[tensor(num_rnn_layers, batch_size, hidden_size)]: tensor
        """
        timesteps, batch_size, _ = hist.shape
      
        KLD = zeros(1).to(self.device)
        NLL = zeros(1).to(self.device)
        MSE = zeros(1).to(self.device)
        h = Variable(
            zeros(self.num_layers, batch_size, self.rnn_dim)).to(self.device)
        context = kwargs.get('context')
        
        for t in range(1, timesteps):
            # extract location features
            x_t = hist[t]
            f_x_t = self.f_x(x_t) 
            
            # extract the context
            f_c_t = None if context is None else self.f_c(context[t])
            
            # patternnet - predict future pattern 
            p_tm1 = pat[t-1].flatten(start_dim=1)
            p_embed = cat(
                [p_tm1, h[-1]], 1) if f_c_t is None else cat([p_tm1, f_c_t, h[-1]], 1)
            p_t_dec = self.dec_pat(p_embed).view(-1, self.pat_len, self.dim)
            
            # patternnet - extract pattern features 
            p_t = pat[t]
            f_p_t = self.f_pat(p_t.flatten(start_dim=1)).squeeze(1)
            
            # c-vae - encoder
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
                [f_z_t, f_p_t, h[-1]], 1) if f_c_t is None else cat(
                    [f_z_t, f_p_t, f_c_t, h[-1]], 1)
            x_dec_t = self.dec(x_dec_embed)
            x_dec_mean_t = x_dec_t[:, :self.dim]
            x_dec_logvar_t = x_dec_t[:, self.dim:]

            # rnn
            h_embed = cat([f_x_t, f_z_t, f_c_t], 1).unsqueeze(0)
            _, h = self.rnn(h_embed, h)

            # compute losses
            KLD += self._kld(
                x_enc_mean_t, x_enc_logvar_t, x_prior_mean_t, x_prior_logvar_t)
            NLL += self._nll_gauss(x_dec_mean_t, x_dec_logvar_t, x_t)
            MSE += self._mse(p_t, p_t_dec)

        return KLD, NLL, MSE

    @torch.no_grad()
    def evaluate(
        self, hist: tensor, pat: tensor, **kwargs
    ) -> Tuple[tensor, tensor, Variable]:
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
        MSE = zeros(1).to(self.device)
        
        h = Variable(zeros(
            self.num_layers, batch_size, self.rnn_dim)).to(self.device)
        context = kwargs.get('context')
        p_tm1 = pat[0].flatten(start_dim=1)
        
        for t in range(1, timesteps):
            # extract location features
            x_t = hist[t]
            f_x_t = self.f_x(x_t) 
            
            # extract the context
            f_c_t = None if context is None else self.f_c(context[t])
            
            # patternnet - predict future pattern 
            p_embed = cat(
                [p_tm1, h[-1]], 1) if f_c_t is None else cat([p_tm1, f_c_t, h[-1]], 1)
            p_t_dec = self.dec_pat(p_embed)
            
            # patternnet - extract pattern features
            f_p_t = self.f_pat(p_t_dec)
            
            # patternnet - set up variables for the next step 
            p_t = pat[t]
            p_tm1 = p_t_dec
            p_t_dec = p_t_dec.view(-1, self.pat_len, self.dim)
            
            # c-vae - encoder
            x_enc_embed = cat(
                [f_x_t, h[-1]], 1) if f_c_t is None else cat([f_x_t, f_c_t, h[-1]], 1)
            x_enc_t = self.enc(x_enc_embed)
            x_enc_mean_t = x_enc_t[:, :self.z_dim]
            x_enc_logvar_t = x_enc_t[:, self.z_dim:]

            # c-vae - prior
            x_prior_embed = h[-1] if f_c_t is None else cat([f_c_t, h[-1]], 1)
            x_prior_t = self.prior(x_prior_embed)
            x_prior_mean_t = x_prior_t[:, :self.z_dim]
            x_prior_logvar_t = x_prior_t[:, self.z_dim:]

            # cvae - sample latent space
            z_t = self._reparameterize(x_enc_mean_t, x_enc_logvar_t)
            
            # extract latent space features
            f_z_t = self.f_z(z_t)

            # cvae - decoder
            x_dec_embed = cat(
                [f_z_t, f_p_t, h[-1]], 1) if f_c_t is None else cat(
                    [f_z_t, f_p_t, f_c_t, h[-1]], 1)
            x_dec_t = self.dec(x_dec_embed)
            x_dec_mean_t = x_dec_t[:, :self.dim]
            x_dec_logvar_t = x_dec_t[:, self.dim:]

            # rnn
            h_embed = cat([f_x_t, f_z_t, f_c_t], 1).unsqueeze(0)
            _, h = self.rnn(h_embed, h)

            # compute losses
            KLD += self._kld(
                x_enc_mean_t, x_enc_logvar_t, x_prior_mean_t, x_prior_logvar_t)
            NLL += self._nll_gauss(x_dec_mean_t, x_dec_logvar_t, x_t)
            MSE += self._mse(p_t, p_t_dec)

        return KLD, NLL, MSE, h, p_t_dec
    
    @torch.no_grad()
    def inference(self, fut_len: int, h: Variable, pat: tensor, **kwargs) -> tensor:
        """ Inference (sampling) trajectories.
        
        Inputs:
        -------
        fut_len[int]: length of the predicted trajectory
        h[torch.Variable(rnn_layers, batch_size, dim)]: torch.Variable 
        pat[tensor(batch_size, pat_len * dim)]
        kwargs: any other keyword-based arguments
        
        Outputs:
        --------
        sample[tensor(fut_len, batch_size, dims)]: predicted trajectories
        """
        _, batch_size, _ = h.shape

        samples = zeros(fut_len, batch_size, self.dim).to(self.device)
        context = kwargs.get('context')
        p_tm1 = pat.flatten(start_dim=1)
        
        for t in range(fut_len):
            # extract the context
            f_c_t = None if context is None else self.f_c(context[t])
            
            # patternnet - predict future pattern 
            p_embed = cat(
                [p_tm1, h[-1]], 1) if f_c_t is None else cat([p_tm1, f_c_t, h[-1]], 1)
            p_t_dec = self.dec_pat(p_embed)
            
            # patternnet - set up variables for the next step 
            p_tm1 = p_t_dec.flatten(start_dim=1)
            
            # patternnet - extract pattern features
            f_p_t = self.f_pat(p_tm1)
            
            # cvae - prior
            x_prior_embed = h[-1] if f_c_t is None else cat([f_c_t, h[-1]], 1)
            x_prior_t = self.prior(x_prior_embed)
            x_prior_mean_t = x_prior_t[:, :self.z_dim]
            x_prior_logvar_t = x_prior_t[:, self.z_dim:]

            # cvae - sample latent space
            z_t = self._reparameterize(x_prior_mean_t, x_prior_logvar_t)
            
            # extract latent space features
            f_z_t = self.f_z(z_t)

            # cvae - decoder 
            x_dec_embed = cat(
                [f_z_t, f_p_t, h[-1]], 1) if f_c_t is None else cat(
                    [f_z_t, f_p_t, f_c_t, h[-1]], 1)
            x_dec_t = self.dec(x_dec_embed)
            x_dec_mean_t = x_dec_t[:, :self.dim]
            
            # add trajectory sample
            samples[t] = x_dec_mean_t.data

            # extract location features from decoded latent space (~ 'x')
            f_x_t = self.f_x(x_dec_mean_t)

            # rnn
            h_embed = cat([f_x_t, f_z_t, f_c_t], 1).unsqueeze(0)
            _, h = self.rnn(h_embed, h)
        
        return samples