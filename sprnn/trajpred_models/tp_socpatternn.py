# ------------------------------------------------------------------------------
# @file:    tp_socpatternn.py
# @brief:   This class extends the TrajPredVRNN class with a pattern learning
#           module and social encoding. 
#           Code based on: sprnn.trajpred_models.tp_vrnn.TrajPredVRNN
# ------------------------------------------------------------------------------
import json
import torch
import torch.nn as nn

from torch import tensor, cat, zeros
from torch.autograd import Variable
from typing import Any, Tuple

from sprnn.trajpred_models.modeling.mlp import MLP
from sprnn.trajpred_models.modeling.mha import MHA
from sprnn.trajpred_models.tp_vrnn import VRNN
from sprnn.utils.common import (
    dotdict, convert_rel_to_abs, compute_social_influences)

class InteractionNet(nn.Module):
    """ A class that implements a model for attending to and encoding social 
    distances between agents. """
    def __init__(self, config: dict, logger: Any, device: str = "cpu") -> None:
        """ Initializes the interaction model.
        
        Inputs:
        -------
        config[dict]: configuration parameters for the model.
        device[str]: device used by the module. By default it uses cpu
        """
        self._name = self.__class__.__name__
        super(InteractionNet, self).__init__()
        
        self._config = config
        logger.info("{} configuration:\n{}".format(
            self.name, json.dumps(self.config, indent=2)
        ))
        
        self.device = device
        logger.info(f"{self.name} uses torch.device({self.device})")
        
        # interaction - multi-head attention 
        mha = dotdict(self.config.interaction_att)
        self.attm = MHA(mha, logger, self.device)
        
        # interaction - down-projection
        # projection = dotdict(self.config.interaction_proj)
        # self.proj = MLP(projection, logger, self.device)
        
        logger.info(f"{self.name} architecture:\n{self}")
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def config(self)-> dict:
        return self._config
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        """ Forward propagation.
        Inputs:
        -------
        x[torch.tensor(seq_len, num_agents, num_agents)]: current agent-to-agents
            distance or displacement tensor. 
            
        Outputs:
        --------
        x[torch.tensor()]: distance features after running Multi-Head Attention 
            and down-projecting the attended features. 
        """
        # attend over extracted features
        x, attention = self.attm(x, return_attention=True)
        x = x.flatten(start_dim=1)
        
        # down-project attended features
        # x = self.proj(x)
        
        # return x, attention
        return x
    
class SocialPatteRNN(VRNN):
    """ A class that implements trajectory prediction model using a VRNN. """
    def __init__(self, config: dict, logger: Any, device: str = "cuda:0") -> None:
        """ Initializes the trajectory prediction network.
        
        Inputs:
        -------
        config[dict]: dictionary containing all configuration parameters.
        device[str]: device name used by the module. By default uses cuda:0. 
        """
        # Intitializes base the base model which is a VRNN module
        super().__init__(config, logger, device)
        
        # PatternNet:
        self.scale = config.scale
        
        pattern_net = dotdict(self.config.pattern_net)
        self.pat_len = pattern_net.pat_len
        self.with_soc_feature = pattern_net.with_soc_feature
        
        # pattern_net - feature extractor
        feat_pat = dotdict(pattern_net.feat_pat)
        feat_pat.in_size = self.pat_len * self.dim 
        self.f_pat = MLP(feat_pat, logger, self.device)
        
        # pattern_net - pattern decoder 
        dec_pat = dotdict(pattern_net.dec_pat)
        dec_pat.out_size = self.pat_len * self.dim
        self.dec_pat = MLP(dec_pat, logger, self.device)
        
        # InteractionNet:
        interaction_net = dotdict(self.config.interaction_net)
        self.k_nearest = interaction_net.k_nearest
        self.type = interaction_net.type 
        
        # interaction_net - feature extractor
        if self.type == "mlp":
            interaction_enc = dotdict(interaction_net.feat_soc)
            self.interaction_net = MLP(interaction_enc, logger, self.device)
            self.flatten = True
        elif self.type == "mha":
            self.interaction_net = InteractionNet(
                interaction_net, logger, self.device)
            self.flatten = False
        else:
            raise NotImplementedError(f"InteractionNet type {self.type}")
        
        logger.info(f"{self.name} architecture:\n{self}")

    def forward(
        self, hist: tensor, pat: tensor, soc: tensor, **kwargs
    ) -> Tuple[tensor, tensor, tensor]:
        """ Forward propagation of observed trajectories.
        
        Inputs:
        -------
        hist[tensor(hist_len, batch_size, dims)]: trajectory histories 
        pat[tensor(hist_len, batch_size, pat_len, dims)]: patterns
        soc[tensor(hist_len, batch_size, soc_len)]: social influences
        kwargs: keyword-based arguments
            
        Outputs:
        --------
        KLD[tensor(1)]: accumulated KL divergence values
        NLL[tensor(1)]: accumulated Neg Log-Likelyhood values
        MSE[tensor(1)]: accumulated Mean Squared Error values
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
            
            # interactionnet - extract social influence features
            s_tm1 = soc[t-1]
            f_s_tm1 = self.interaction_net(s_tm1)
            s_tm1 = s_tm1.flatten(start_dim=1)
            
            # patternnet - predict future pattern 
            p_tm1 = pat[t-1].flatten(start_dim=1)
            if self.with_soc_feature:
                p_embed = cat(
                    [p_tm1, f_s_tm1, h[-1]], 1) if f_c_t is None else cat(
                        [p_tm1, f_s_tm1, f_c_t, h[-1]], 1)
            else: 
                p_embed = cat(
                    [p_tm1, s_tm1, h[-1]], 1) if f_c_t is None else cat(
                        [p_tm1, s_tm1, f_c_t, h[-1]], 1)
            p_t_dec = self.dec_pat(p_embed).view(-1, self.pat_len, self.dim)
            
            # patternnet - extract pattern features 
            p_t = pat[t]
            f_p_t = self.f_pat(p_t.flatten(start_dim=1)).squeeze(1)
            
            # c-vae - encoder
            x_enc_embed = cat(
                [f_x_t, f_s_tm1, h[-1]], 1) if f_c_t is None else cat(
                    [f_x_t, f_s_tm1, f_c_t, h[-1]], 1)
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
            
            # extract latent space features
            f_z_t = self.f_z(z_t)

            # cvae - decoder
            x_dec_embed = cat(
                [f_z_t, f_p_t, f_s_tm1, h[-1]], 1) if f_c_t is None else cat(
                    [f_z_t, f_p_t, f_s_tm1, f_c_t, h[-1]], 1)
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
        self, hist: tensor, hist_abs: tensor, pat: tensor, soc: tensor, 
        seq_start_end: tensor, **kwargs
    ) -> Tuple[tensor, tensor, tensor, tensor, tensor]:
        """ Forward propagation of observed trajectories.
        
        Inputs:
        -------
        hist[tensor(hist_len, batch_size, dims)]: trajectory histories (rel)
        hist_abs[tensor(hist_len, batch_size, dims)]: trajectory histories (abs)
        pat[tensor(hist_len, batch_size, pat_len, dim)]: patterns
        soc[tensor(hist_len, batch_size, k_nearest * dim)]: social influences
        seq_start_end[tensor]: tensor indicating all sequences starts and ends. 
        kwargs: keyword-based arguments
            
        Outputs:
        --------
        KLD[tensor]: accumulated KL divergence values
        NLL[tensor]: accumulated Neg Log-Likelyhood values
        MSE[tensor]: accumulated Mean Squared Error values
        h[tensor(num_rnn_layers, batch_size, r)]: tensor
        """
        timesteps, batch_size, _ = hist.shape
      
        KLD = zeros(1).to(self.device)
        NLL = zeros(1).to(self.device)
        MSE = zeros(1).to(self.device)
        
        h = Variable(
            zeros(self.num_layers, batch_size, self.rnn_dim).to(self.device))
        context = kwargs.get('context')
        p_tm1 = pat[0].flatten(start_dim=1)
        s_tm1 = soc[0]
        
        for t in range(1, timesteps):
            # extract location features
            x_t = hist[t]
            f_x_t = self.f_x(x_t) 
            
            # extract the context
            f_c_t = None if context is None else self.f_c(context[t])
            
            # interactionnet - extract social influence features
            f_s_tm1 = self.interaction_net(s_tm1)
            s_tm1 = s_tm1.flatten(start_dim=1)
            
            # patternnet - predict future pattern 
            if self.with_soc_feature:
                p_embed = cat(
                    [p_tm1, f_s_tm1, h[-1]], 1) if f_c_t is None else cat(
                        [p_tm1, f_s_tm1, f_c_t, h[-1]], 1)
            else: 
                p_embed = cat(
                    [p_tm1, s_tm1, h[-1]], 1) if f_c_t is None else cat(
                        [p_tm1, s_tm1, f_c_t, h[-1]], 1)
            p_t_dec = self.dec_pat(p_embed)
            
            # patternnet - extract pattern features
            f_p_t = self.f_pat(p_t_dec)
            
            # patternnet - set up variables for the next step 
            p_t = pat[t]
            p_tm1 = p_t_dec
            p_t_dec = p_t_dec.view(-1, self.pat_len, self.dim)
            
            # c-vae - encoder
            x_enc_embed = cat(
                [f_x_t, f_s_tm1, h[-1]], 1) if f_c_t is None else cat(
                    [f_x_t, f_s_tm1, f_c_t, h[-1]], 1)
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
                [f_z_t, f_p_t, f_s_tm1, h[-1]], 1) if f_c_t is None else cat(
                    [f_z_t, f_p_t, f_s_tm1, f_c_t, h[-1]], 1)
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
            
            # compute the new social influences
            x_abs_t = hist_abs[t]
            p_abs_t = convert_rel_to_abs(p_t_dec, x_abs_t)
            s_tm1 = compute_social_influences(
                x_abs_t.unsqueeze(0).detach().cpu(), 
                p_abs_t[:, -1].unsqueeze(0).detach().cpu(), 
                seq_start_end, self.k_nearest, self.flatten).to(self.device)[0]
        
        return KLD, NLL, MSE, h, p_tm1, s_tm1
    
    @torch.no_grad()
    def inference(
        self, x_abs: tensor, fut_len: int, h: Variable, pat: tensor, soc: tensor, 
        seq_start_end: tensor, coord: str, **kwargs
    ) -> tensor:
        """ Inference (sampling) trajectories.
        
        Inputs:
        -------
        x_abs[tensor(1, batch_size, dim)]: last step from observed history.
        fut_len[int]: length of the predicted trajectory.
        h[tensor(rnn_layers, batch_size, dim)]: last hidden state. 
        pat[tensor(batch_size, pat_len * dim)]: last pattern.
        soc[tensor(batch_size, knearest * dim)]: last social influence.
        seq_start_end[tensor]: tensor indicating all sequences starts and ends. 
        kwargs: any other keyword-based arguments
        
        Outputs:
        --------
        sample[tensor(fut_len, batch_size, dims)]: predicted trajectories
        """
        _, batch_size, _ = h.shape

        p_tm1 = pat
        s_tm1 = soc
        x_abs_t = x_abs
        context = kwargs.get('context')
        
        samples = zeros(fut_len, batch_size, self.dim).to(self.device)
        
        for t in range(fut_len):
            # extract the context
            f_c_t = None if context is None else self.f_c(context[t])

            # interactionnet - extract social influence features
            f_s_tm1 = self.interaction_net(s_tm1)
            s_tm1 = s_tm1.flatten(start_dim=1)
            
            # patternnet - predict future pattern 
            if self.with_soc_feature:
                p_embed = cat(
                    [p_tm1, f_s_tm1, h[-1]], 1) if f_c_t is None else cat(
                        [p_tm1, f_s_tm1, f_c_t, h[-1]], 1)
            else: 
                p_embed = cat(
                    [p_tm1, s_tm1, h[-1]], 1) if f_c_t is None else cat(
                        [p_tm1, s_tm1, f_c_t, h[-1]], 1)
            p_t_dec = self.dec_pat(p_embed)
            
            # patternnet - extract pattern features
            f_p_t = self.f_pat(p_t_dec)
            
            # patternnet - set up variables for the next step 
            p_tm1 = p_t_dec
            p_t_dec = p_t_dec.view(-1, self.pat_len, self.dim)
            
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
                [f_z_t, f_p_t, f_s_tm1, h[-1]], 1) if f_c_t is None else cat(
                    [f_z_t, f_p_t, f_s_tm1, f_c_t, h[-1]], 1)
            x_dec_t = self.dec(x_dec_embed)
            x_dec_mean_t = x_dec_t[:, :self.dim]
            
            # add trajectory sample
            samples[t] = x_dec_mean_t.data

            # extract location features from decoded latent space (~ 'x')
            f_x_t = self.f_x(x_dec_mean_t)

            # rnn
            h_embed = cat([f_x_t, f_z_t, f_c_t], 1).unsqueeze(0)
            _, h = self.rnn(h_embed, h)
            
            # compute new social influence
            if coord == "rel":
                x_abs_t = x_abs_t + x_dec_mean_t
            else:
                x_abs_t = x_dec_mean_t
            p_abs_t = convert_rel_to_abs(p_t_dec, x_abs_t)
            s_tm1 = compute_social_influences(
                x_abs_t.unsqueeze(0).detach().cpu(), 
                p_abs_t[:, -1].unsqueeze(0).detach().cpu(), 
                seq_start_end, self.k_nearest, self.flatten).to(self.device)[0]
            
        return samples