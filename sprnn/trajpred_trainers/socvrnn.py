# ------------------------------------------------------------------------------
# @file:    socvrnn.py
# @brief:   This file contains the implementation of the SocialVRNNTrainer class 
#           is used for training a Variational Recurrent Neural Network for 
#           trajectory prediction with a vanilla module for "social" encoding. 
# ------------------------------------------------------------------------------
import os
from time import time
import torch
import torch.optim as optim

from tqdm import tqdm 

import sprnn.utils.common as mutils

from sprnn.trajpred_models.tp_socvrnn import SocialVRNN
from sprnn.trajpred_trainers.base_trainer import BaseTrainer
from sprnn.utils import metrics

class SocialVRNNTrainer(BaseTrainer):
    """ A trainer class that implements trainer methods for the VRNN trajectory 
    prediction model. It inherits base methods for training and evaluating from 
    the BaseTrainer (See sprnn/trajpred_trainers/base_trainer.py). """
    def __init__(self, config: dict) -> None:
        """ Initializes the trainer.
        
        Inputs:
        -------
        config[dict]: a dictionary containing all configuration parameters.
        """
        super().__init__(config)
        self.setup()
        self.logger.info(f"{self.name} is ready!")
        
    def train_epoch(self, epoch: int) -> dict:
        """ Trains one epoch.
        
        Inputs:
        -------
        epoch[int]: current training epoch
             
        Outputs:
        --------
        loss[dict]: a dictionary with all loss values computed during the epoch.
        """
        epoch_str = f"[{epoch}/{self.num_epoch}]"
        self.logger.info(f"Training epoch: {epoch_str}")
        self.model.train()
        
        batch_count = 0 
        batch_loss = 0

        self.train_losses.reset()
        pbar = tqdm(total=self.num_iter, leave=False)
        
        for i, batch in enumerate(self.train_data):
            if i >= self.num_iter:
                break
            self.optimizer.zero_grad()

            # NOTE:
            # hist_abs, hist_rel have shapes (hist_len, num_agents, dim)
            # fut_abs, fut_rel have shapes(fut_len, num_agents, dim)
            batch = [tensor.to(self.device) for tensor in batch]   
            (hist_abs, pat_abs, fut_abs, hist_rel, pat_rel, fut_rel, weather, 
             seq_start_end) = batch
            
            timesteps, batch_size, dim = hist_rel.shape       
            
            # TODO: make this work for variable number of agents
            if self.coord == "rel":
                hist = hist_rel.view(timesteps, -1, 2 * dim)
            else: # abs coords
                hist = hist_abs.view(timesteps, -1, 2 * dim)
            kld, nll = self.model(hist)
            
            loss = self.compute_loss(epoch=epoch, kld=kld, nll=nll)
            batch_loss += loss['Loss']
            batch_count += 1
    
            if batch_count >= self.batch_size:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.gradient_clip)
                self.optimizer.step()
                batch_loss = 0.0
                batch_count = 0
                
            self.train_losses.update([loss], batch_size)
            pbar.update(1)
            
        return self.train_losses.get_losses()

    @torch.no_grad()
    def eval_epoch(self, epoch: int, **kwargs) -> dict:
        """ Evaluates one epoch.
        Inputs:
        -------
        epoch[int]: current eval epoch.
        **kwargs: keyword arguments as needed by the trainer. 
        
        Outputs:
        --------
        loss[dict]: a dictionary with all of losses computed during training. 
        """  
        self.model.eval()
        num_samples = kwargs.get('num_samples') if kwargs.get('num_samples') else 1
        
        self.eval_losses.reset()
        self.eval_metrics.reset()
        
        pbar = tqdm(total=self.eval_num_iter, leave=False)
        
        for i, batch in enumerate(self.val_data):                   
            if i >= self.eval_num_iter:
                break
           
            batch = [tensor.to(self.device) for tensor in batch]
            (hist_abs, pat_abs, fut_abs, hist_rel, pat_rel, fut_rel, weather, 
             seq_start_end) = batch

            timesteps, batch_size, dim = hist_rel.shape       
            
            # TODO: make this work for variable number of agents
            if self.coord == "rel":
                hist = hist_rel.view(timesteps, -1, 2 * dim)
            else: # abs coords
                hist = hist_abs.view(timesteps, -1, 2 * dim)
            kld, nll, h_H = self.model.evaluate(hist)
            
            loss = self.compute_loss(epoch=epoch, kld=kld, nll=nll)
                    
            # generate future steps
            pred_list = []
            x_H = hist_abs[-1]
            for _ in range(num_samples):
                h = h_H.clone()
                
                # run inference to predict the trajectory's future steps
                pred = self.model.inference(
                    self.fut_len, h).view(self.fut_len, -1, dim)
                
                if self.coord == "rel":    
                    # convert the prediction to absolute coords
                    pred = mutils.convert_rel_to_abs(pred, x_H, permute=True)
                    
                pred_list.append(pred)
                  
            # compute best of num_samples
            preds = torch.stack(pred_list)
            _ = self.eval_metrics.update(fut_abs, preds, seq_start_end)
            metrics = self.eval_metrics.get_metrics()
            
            self.eval_losses.update([loss], batch_size)
            losses = self.eval_losses.get_losses()
            metrics.update(losses)
            pbar.update(1)
                                    
        return metrics
    
    @torch.no_grad()
    def test_epoch(self, epoch: int, **kwargs) -> dict:
        """ Tests one epoch.
        Inputs:
        -------
        epoch[int]: current eval epoch.
        **kwargs: keyword arguments as needed by the trainer. 
        
        Outputs:
        --------
        loss[dict]: a dictionary with all of losses computed during training. 
        """  
        self.model.eval()
        num_samples = kwargs.get('num_samples') if kwargs.get('num_samples') else 1
        
        self.eval_losses.reset()
        self.eval_metrics.reset()
        
        pbar = tqdm(total=self.test_num_iter, leave=False)
        
        for i, batch in enumerate(self.test_data):                   
            if i >= self.test_num_iter:
                break
           
            batch = [tensor.to(self.device) for tensor in batch]
            (hist_abs, pat_abs, fut_abs, hist_rel, pat_rel, fut_rel, weather, 
             seq_start_end) = batch

            timesteps, batch_size, dim = hist_rel.shape       
            
            # TODO: make this work for variable number of agents
            if self.coord == "rel":
                hist = hist_rel.view(timesteps, -1, 2 * dim)
            else: # abs coords
                hist = hist_abs.view(timesteps, -1, 2 * dim)
            kld, nll, h_H = self.model.evaluate(hist)
            
            loss = self.compute_loss(epoch=epoch, kld=kld, nll=nll)
                    
            # generate future steps
            pred_list = []
            x_H = hist_abs[-1]
            for _ in range(num_samples):
                h = h_H.clone()
                
                # run inference to predict the trajectory's future steps
                pred = self.model.inference(
                    self.fut_len, h).view(self.fut_len, -1, dim)
                
                if self.coord == "rel":    
                    # convert the prediction to absolute coords
                    pred = mutils.convert_rel_to_abs(pred, x_H, permute=True)
                    
                pred_list.append(pred)
                  
            # compute best of num_samples
            preds = torch.stack(pred_list)
            _ = self.eval_metrics.update(fut_abs, preds, seq_start_end)
            metrics = self.eval_metrics.get_metrics()
            
            self.eval_losses.update([loss], batch_size)
            losses = self.eval_losses.get_losses()
            metrics.update(losses)
            pbar.update(1)
                                    
        return metrics
    
    @torch.no_grad()
    def eval_sample(self, hist_abs):
        # import pdb; pdb.set_trace()
        timesteps, batchsize, dim = hist_abs.shape
        hist_abs = hist_abs.to(self.device).reshape(timesteps, 1, 2 * dim)
        
        kld, nll, h = self.model.evaluate(hist_abs)
                        
        # run inference to predict the trajectory's future steps
        pred = self.model.inference(self.fut_len, h).view(self.fut_len, -1, dim)
        
        return pred.cpu()
        
    def setup(self) -> None:
        """ Sets the trainer as follows:
            * model: VRNN
            * optimizer: AdamW
            * lr_scheduler: ReduceOnPlateau 
        """
        self.logger.info(f"{self.name} setting up model: {self.trainer}")

        self.model = SocialVRNN(
            self.config.MODEL, self.logger, self.device).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.config.TRAIN.lr)
        
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, threshold=1e-2, patience=10, factor=5e-1, verbose=True)
        
        self.train_losses = metrics.LossContainer()
        self.eval_losses = metrics.LossContainer()
        self.eval_metrics = metrics.MetricContainer()
        self.main_metric = self.eval_metrics.get_main_metric()
        
        if self.config.TRAIN.load_model:
            ckpt_file = os.path.join(self.out.ckpts, self.config.TRAIN.ckpt_name)
            assert os.path.exists(ckpt_file), f"Ckpt {ckpt_file} does not exist!"
            self.load_model(ckpt_file)