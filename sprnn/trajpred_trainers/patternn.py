# ------------------------------------------------------------------------------
# @file:    patternn.py
# @brief:   This file contains the implementation of the PatteRNNTrainer class
#           which is used for training a Variational Recurrent Neural Network
#           with pattern learning for trajectory prediction. 
# ------------------------------------------------------------------------------
import os
import torch
import torch.optim as optim

from tqdm import tqdm 

import sprnn.utils.common as mutils

from sprnn.trajpred_models.tp_patternn import PatteRNN
from sprnn.trajpred_trainers.base_trainer import BaseTrainer
from sprnn.utils import metrics

class PatteRNNTrainer(BaseTrainer):
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
            (hist_abs, pat_abs, fut_abs, hist_rel, pat_rel, fut_rel, context, 
             seq_start_end) = batch
            
            batch_size = hist_rel.shape[1]        
            pat_rel = pat_rel[:, :, :, :self.dim]
        
            # import pdb; pdb.set_trace()
            # pat_abs_rec = mutils.convert_rel_to_abs(pat_rel[0], hist_abs[0])
            
            if self.coord == "rel":
                hist_rel = hist_rel[:, :, :self.dim]
                kld, nll, mse = self.model(hist_rel, pat_rel, context=context)
            else:
                hist_abs = hist_abs[:, :, :self.dim]
                kld, nll, mse = self.model(hist_abs, pat_rel, context=context)
                
            loss = self.compute_loss(epoch=epoch, kld=kld, nll=nll, mse=mse)
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
        self.eval_losses.reset()
        self.eval_metrics.reset()
        
        pbar = tqdm(total=self.eval_num_iter, leave=False)
        
        for i, batch in enumerate(self.val_data):                   
            if i >= self.eval_num_iter:
                break
           
            batch = [tensor.to(self.device) for tensor in batch]
            (hist_abs, pat_abs, fut_abs, hist_rel, pat_rel, fut_rel, context, 
             seq_start_end) = batch
            
            batch_size = hist_abs.shape[1]
            
            # constrain trajectories to specified number of dims
            hist_abs = hist_abs[:, :, :self.dim]
            hist_rel = hist_rel[:, :, :self.dim]
            fut_abs = fut_abs[:, :, :self.dim]
            pat_rel = pat_rel[:, :, :, :self.dim]
            
            # eval burn-in process 
            if self.coord == "rel":
                kld, nll, mse, h_H, pat_H = self.model.evaluate(
                    hist_rel, pat_rel, context=context)
            else:
                kld, nll, mse, h_H, pat_H = self.model.evaluate(
                    hist_abs, pat_rel, context=context)
            
            loss = self.compute_loss(epoch=epoch, kld=kld, nll=nll, mse=mse)
            
            # generate future steps
            pred_list = []
            x_H = hist_abs[-1]
            for _ in range(self.num_samples):
                h, pat = h_H.clone(), pat_H.clone()

                # run inference to predict the trajectory's future steps
                pred = self.model.inference(self.fut_len, h, pat, context=context)
                
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
        self.eval_losses.reset()
        self.eval_metrics.reset()
        
        pbar = tqdm(total=self.test_num_iter, leave=False)
        
        for i, batch in enumerate(self.test_data):                   
            if i >= self.test_num_iter:
                break
           
            batch = [tensor.to(self.device) for tensor in batch]
            (hist_abs, pat_abs, fut_abs, hist_rel, pat_rel, fut_rel, context, 
             seq_start_end) = batch
            
            batch_size = hist_abs.shape[1]
            
            # constrain trajectories to specified number of dims
            hist_abs = hist_abs[:, :, :self.dim]
            hist_rel = hist_rel[:, :, :self.dim]
            fut_abs = fut_abs[:, :, :self.dim]
            pat_rel = pat_rel[:, :, :, :self.dim]
            
            # eval burn-in process 
            if self.coord == "rel":
                kld, nll, mse, h_H, pat_H = self.model.evaluate(
                    hist_rel, pat_rel, context=context)
            else:
                kld, nll, mse, h_H, pat_H = self.model.evaluate(
                    hist_abs, pat_rel, context=context)
            
            loss = self.compute_loss(epoch=epoch, kld=kld, nll=nll, mse=mse)
            
            # generate future steps
            pred_list = []
            x_H = hist_abs[-1]
            for _ in range(self.num_samples):
                h, pat = h_H.clone(), pat_H.clone()

                # run inference to predict the trajectory's future steps
                pred = self.model.inference(self.fut_len, h, pat, context=context)
                
                if self.coord == "rel":
                    # convert the prediction to absolute coords
                    pred = mutils.convert_rel_to_abs(pred, x_H, permute=True)
                    
                pred_list.append(pred)
                    
            # compute best of num_samples
            preds = torch.stack(pred_list)
            best_sample_idx = self.eval_metrics.update(
                fut_abs, preds, seq_start_end)
            metrics = self.eval_metrics.get_metrics()
            
            self.eval_losses.update([loss], batch_size)
            losses = self.eval_losses.get_losses()
            metrics.update(losses)
            pbar.update(1)
            
            if self.visualize and i % self.plot_freq == 0:
                self.generate_outputs(
                    hist_abs, fut_abs, preds, best_sample_idx, seq_start_end, 
                    f"epoch-{epoch+1}_test-{i}")
                        
        return metrics
    
    @torch.no_grad()
    def eval_sample(self, hist_abs):
        # import pdb; pdb.set_trace()
        # shape = (hist_len, num_agents, dim)
        H, N, D = hist_abs.shape
        hist_abs = hist_abs.to(self.device)
        patterns = torch.zeros((H, N, self.pat_len-1, D)).to(self.device)
        pat = hist_abs[:self.pat_len]
        patterns[0] = torch.transpose(pat[1:] - pat[:-1], 0, 1).to(self.device)
        _, _, _, h, pat = self.model.evaluate(hist_abs, patterns)
        
        # repeat hidden state and final pattern self.num_samples times to 
        # parallelize inference
        h = h.repeat(1, self.num_samples, 1)
        pat = pat.repeat(self.num_samples, 1, 1)
         
        # run inference to predict the trajectory's future steps
        pred = self.model.inference(self.fut_len, h, pat)
        return pred.cpu()
        
    def setup(self) -> None:
        """ Sets the trainer as follows:
            * model: TrajPredVRNN
            * optimizer: AdamW
            * lr_scheduler: ReduceOnPlateau 
        """
        self.logger.info(f"{self.name} setting up model: {self.trainer}")
    
        self.model = PatteRNN(
            self.config.MODEL, self.logger, self.device).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.config.TRAIN.lr)
        
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, threshold=1e-2, patience=50, factor=5e-1, verbose=True)
        
        self.train_losses = metrics.LossContainer(
            loss_list=['Loss', 'LossKLD', 'LossNLL', 'LossMSE'])
        self.eval_losses = metrics.LossContainer(
            loss_list=['Loss', 'LossKLD', 'LossNLL', 'LossMSE'])
        self.eval_metrics = metrics.MetricContainer()
        self.main_metric = self.eval_metrics.get_main_metric()
        
        if self.config.TRAIN.load_model:
            ckpt_file = os.path.join(self.out.ckpts, self.config.TRAIN.ckpt_name)
            assert os.path.exists(ckpt_file), f"Ckpt {ckpt_file} does not exist!"
            self.load_model(ckpt_file)