# ------------------------------------------------------------------------------
# @file:    metrics.py
# @brief:   This file contains the implementation of the measure and loss classes
#           class which for computing and containing all metrics needed for 
#           evaluation. 
# ------------------------------------------------------------------------------
import torch

class LossContainer:
    """ A class for maintaining losses """
    def __init__(self, loss_list = ['Loss', 'LossKLD', 'LossNLL']) -> None:
        # initialize 
        self.losses = dict()
        for loss in loss_list:
            self.losses[loss] = Loss()
    
    def add_loss(self, loss_name: str) -> None:
        self.losses[loss_name] = Loss()
          
    def reset(self) -> None:
        for k, v in self.losses.items():
            v.reset()
            
    def update(self, losses, batch_size):
        for loss in losses:
            for k, v in loss.items():
                v = v.item() if torch.is_tensor(v) else v
                if not self.losses.get(k):
                    continue
                self.losses[k].compute(batch_size=batch_size, value=v)
    
    def get_losses(self) -> dict:
        
        losses = {}
        for k, v in self.losses.items():
            losses[k] = v.get_metric()
        return losses

class MetricContainer:
    """ A class for maintaining metric """
    def __init__(
        self, metric_list = ['MinADE', 'MinFDE', 'ADE', 'FDE'], 
        main_metric = 'MinADE') -> None:
        if len(metric_list) == 1:
            main_metric = metric_list[0]
        
        assert main_metric in metric_list, f"{main_metric} not in {metric_list}"
        self.main_metric = main_metric
        
        # initialize 
        self.metrics = dict()
        for metric in metric_list:
            if metric == 'ADE':
                self.metrics[metric] = ADE(mode='mean')
            elif metric == 'MinADE':
                self.metrics[metric] = ADE(mode='min')
            elif  metric == 'FDE':
                self.metrics[metric] = FDE(mode='mean')
            elif metric == 'MinFDE':
                self.metrics[metric] = FDE(mode='min')
            else:
                raise NotImplementedError(f"Metric {metric} not supported!")

    def reset(self) -> None:
        
        for k, v in self.metrics.items():
            v.reset()
    
    def update(self, traj_gt, traj_pred, seq_start_end, permute=True):
        
        idx = self.metrics[self.main_metric].compute(
            traj_gt=traj_gt, traj_pred=traj_pred, seq_start_end=seq_start_end, 
            permute=permute)

        for k, v in self.metrics.items(): 
            if k == self.main_metric:
                continue
            v.compute(
                traj_gt=traj_gt, traj_pred=traj_pred, seq_start_end=seq_start_end, 
                permute=permute)
            
        return idx
            
    def get_metrics(self) -> dict:
        metrics = {}
        for k, v in self.metrics.items():
            m = v.get_metric()
            m = m.item() if torch.is_tensor(m) else m
            metrics[k] = m
        return metrics
    
    def get_main_metric(self) -> str:
        return self.main_metric
    
class Metric:
    """ Base class for implementing metrics. """
    def __init__(self) -> None:
        """ Initialization. """
        self.metric = 0.0
        self.accum = 0
    
    def reset(self, value = 0.0) -> float:
        self.metric = value
        self.accum = 0
        
    def get_metric(self) -> float:
        if self.accum < 1:
            return 0.0
        return self.metric / self.accum
    
    def compute(self, **kwargs) -> float:
        raise NotImplementedError
    
class Loss:
    """ Base class for implementing losses. """
    def __init__(self) -> None:
        """ Initialization. """
        self.metric = 0.0
        self.accum = 0
    
    def reset(self, value = 0.0) -> float:
        self.metric = value
        self.accum = 0
        
    def get_metric(self) -> float:
        if self.accum < 1:
            return 0.0
        return self.metric / self.accum
    
    def compute(self, **kwargs) -> float:
        self.accum += kwargs.get('batch_size')
        self.metric += kwargs.get('value')
        
class ADE(Metric):
    """ Computes min/mean average displacement error (ADE) of N trajectories. """
    def __init__(self, mode = 'mean') -> None:
        super().__init__()
        
        self.mode = mode
        
    def compute(self, **kwargs):
        permute = True
        if not kwargs.get('permute') is None:
            permute = kwargs.get('permute')
        
        # import pdb;pdb.set_trace()
        # NOTE:
        # traj_gt -> (1, pred_len, batch_size, dim)
        # traj_pred -> (num_samples, pred_len, batch_size, dim)
        
        traj_gt = kwargs.get('traj_gt').unsqueeze(0)
        traj_pred = kwargs.get('traj_pred')
        seq_start_end = kwargs.get('seq_start_end')
        
        assert traj_gt[0].shape == traj_pred[0].shape, \
            f"Shape mismatch: gt {traj_gt.shape} pred {traj_pred.shape}"
        
        seq_len, batch_size, _ = traj_pred[0].shape
        
        if permute:
            # y1 -> (1, batch_size, pred_len, dim)
            y1 = traj_gt.permute(0, 2, 1, 3)
            # y2 -> (num_samples, batch_size, pred_len, dim)
            y2 = traj_pred.permute(0, 2, 1, 3)
        
        # ade -> (num_samples, batch_size)
        ade = torch.sum(torch.sqrt(torch.sum((y1 - y2) ** 2, dim=3)), dim=2)
        
        ade_sum = 0
        for start, end in seq_start_end:
            err = ade[:, start:end]
            err = torch.sum(err, dim=1)
            
            if self.mode == 'min':
                err = torch.min(err)
                ade_sum += err
            elif self.mode == 'mean':
                err = torch.mean(err)
                ade_sum += err
            else:
                raise NotImplementedError(f"Mode: {self.mode}")
        
        idx = None  
        if self.mode == 'min':
            ade_min = torch.min(ade, dim=0)
            ade, idx = ade_min.values, ade_min.indices
            
        self.metric += ade_sum #torch.sum(ade)
        self.accum += batch_size * seq_len
            
        return idx
        
    
class FDE(Metric):
    """ Computes average displacement error (ADE). """
    def __init__(self, mode = 'mean') -> None:
        super().__init__()
        
        self.mode = mode
        
    def compute(self, **kwargs):
        """ Computes and accumulates final displacement error (fde). 
        Inputs:
        ------
        endpoint_gt[torch.tensor]: ground truth final positions with shape (batch, dim).
        endpoint_pred[torch.tensor]: predicted final positions with shape (batch, dim).
        """
        endpoint_gt = kwargs.get('traj_gt').unsqueeze(0)[:, -1]
        endpoint_pred = kwargs.get('traj_pred')[:, -1]
        seq_start_end = kwargs.get('seq_start_end')
        
        assert endpoint_gt[0].shape == endpoint_pred[0].shape, \
            f"Shape mismatch: gt {endpoint_gt.shape} pred {endpoint_pred.shape}"
        
        _, batch_size, _ = endpoint_gt.shape
        
        fde = torch.sqrt(torch.sum((endpoint_gt - endpoint_pred) ** 2, dim=2))
        
        fde_sum = 0
        for start, end in seq_start_end:
            err = fde[:, start:end]
            err = torch.sum(err, dim=1)
            
            if self.mode == 'min':
                err = torch.min(err)
                fde_sum += err
            elif self.mode == 'mean':
                err = torch.mean(err)
                fde_sum += err
            else:
                raise NotImplementedError(f"Mode: {self.mode}")
            
        # if self.mode == 'mean':
        #     fde = torch.mean(fde, dim=0)
        # elif self.mode == 'min': 
        #     fde = torch.min(fde, dim=0).values
    
        self.metric += fde_sum #torch.sum(fde)
        self.accum += batch_size