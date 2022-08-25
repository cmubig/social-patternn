# ------------------------------------------------------------------------------
# @file:    base_trainer.py
# @brief:   This file contains the implementation of the BaseTrainer class used
#           used as the base class for implementing Trajectory Prediction 
#           trainers.  
# ------------------------------------------------------------------------------
import json
import logging
import os
import math
import numpy as np
import random
import torch

from natsort import natsorted
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 

# sprnn modules
import sprnn.utils.common as mutils    

from sprnn.utils import visualization as vis
from sprnn.utils.common import Config, DIMS, COORDS, FORMAT
from sprnn.utils.data_loader import load_data

class BaseTrainer:
    """ A class that implements base trainer methods. """
    def __init__(self, config: dict) -> None:
        """ Initializes the trainer.
        
        Inputs:
        -------
        config[dict]: a dictionary containing all configuration parameters.
        """
        self._config = Config(config)
        super().__init__()
        
        # set random seed
        seed = self.config.TRAIN.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        self.trainer = self.config.BASE_CONFIG.trainer
        self.coord = self.config.BASE_CONFIG.coord
        assert self.coord in COORDS, f"coord {self.coord} not supported ({COORDS})"
        self.dim = self.config.MODEL.dim
        assert self.dim in DIMS, f"Dimension {self.dim} not supported ({DIMS})"

        self.create_output_dirs()
        self.save_config(config, 'config.json')
        
        # loads or generates dataset using config specifications
        train_len = val_len = test_len = 0
        if self.config.DATASET.preprocess:
            self.train_data, self.val_data, self.test_data = load_data(
                self.config.DATASET, self.config.TRAJECTORY)
            train_len, val_len = len(self.train_data), len(self.val_data)
            test_len = len(self.test_data)
        self.logger.info(
            f"Dataset size - train: {train_len} val: {val_len} test: {test_len}")
        
        self.train_batch_size = self.config.DATASET.train_batch_size
        self.val_batch_size = self.config.DATASET.val_batch_size
        self.test_batch_size = self.config.DATASET.test_batch_size
        
        self.device = (
            torch.device("cuda", self.config.BASE_CONFIG.gpu_id)
            if torch.cuda.is_available() and not self.config.BASE_CONFIG.use_cpu
            else torch.device("cpu")
        )
        self.logger.info(f"{self.name} uses torch.device({self.device})")

        # hyper-parameters
        self.batch_size = self.config.TRAIN.batch_size
        self.val_batch_size = self.config.TRAIN.val_batch_size
        self.num_samples = self.config.TRAIN.num_samples

        if self.config.TRAIN.num_iter < 0:
            self.config.TRAIN.num_iter = train_len
        self.num_iter = min(self.config.TRAIN.num_iter, train_len)
        
        if self.config.TRAIN.eval_num_iter < 0:
            self.config.TRAIN.eval_num_iter = val_len
        self.eval_num_iter = min(self.config.TRAIN.eval_num_iter, val_len)
        
        if self.config.TRAIN.test_num_iter < 0:
            self.config.TRAIN.test_num_iter = test_len
        self.test_num_iter = min(self.config.TRAIN.test_num_iter, test_len)

        self.num_epoch = self.config.TRAIN.num_epoch
        
        self.hist_len = int(
            self.config.TRAJECTORY.hist_len / self.config.TRAJECTORY.step)
        self.fut_len = int(
            self.config.TRAJECTORY.fut_len / self.config.TRAJECTORY.step)
        self.traj_len = self.hist_len + self.fut_len
        self.pat_len = self.config.TRAJECTORY.pat_len
        
        self.patience = self.config.TRAIN.patience 
        
        self.warmup = np.ones(self.num_epoch)
        warmup_epochs = self.config.TRAIN.warmup_epochs
        if self.config.TRAIN.warmup:
            self.warmup[:warmup_epochs] = np.linspace(0, 1, num=warmup_epochs)
            
        self.gradient_clip = (
            self.config.TRAIN.gradient_clip if self.config.TRAIN.gradient_clip else 10)
        
        self.update_lr = self.config.TRAIN.update_lr
        self.dataset_name = self.config.DATASET.name
        
        self.max_agents = self.config.TRAIN.max_agents
        self.visualize = self.config.VISUALIZATION.visualize
        self.plot_freq = self.config.VISUALIZATION.plot_freq
        
        self.main_metric = 'MinADE'

    @property
    def config(self) -> Config:
        return self._config

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def train(self, do_eval: bool = False, resume: bool = False) -> None:
        """ Base training process. The method specific to the trainer is 
        train_epoch().
        
        Inputs:
        -------
        do_eval[bool]: if True, it will run eval_epoch() after train_epoch().
        resume[bool]: if True, it will train from the last saved checkpoint. 
        """
        self.logger.info("{} training details:\n{}".format(
            self.name, json.dumps(self.config.TRAIN, indent=2)))

        best_error = float('inf')
        start_epoch = 0
        no_change = 0
        
        # check if start training from a previous checkpoint
        if self.config.TRAIN.load_model:
            ckpt_file = os.path.join(self.out.ckpts, self.config.TRAIN.ckpt_name)
            assert os.path.exists(ckpt_file), f"Ckpt {ckpt_file} does not exist!"

            self.load_model(ckpt_file)

            start_epoch = int(self.config.TRAIN.ckpt_name.split('_')[-1].split('.')[0])
        
        if resume:
            last_ckpt = natsorted(os.listdir(self.out.ckpts))[-1]
            ckpt_file = os.path.join(self.out.ckpts, last_ckpt)
            assert os.path.exists(ckpt_file), f"Ckpt {ckpt_file} does not exist!"

            self.load_model(ckpt_file)
            start_epoch = int(last_ckpt.split('_')[-1].split('.')[0])
            
        # start training
        for epoch in tqdm(range(start_epoch, self.num_epoch)):
            ep = epoch+1
            epoch_str = f"Epoch[{ep}/{self.num_epoch}]\n\ttrain: "
            loss = self.train_epoch(ep)
            # TODO: find better way to handle nan-loss
            if math.isnan(loss['Loss']):
                self.logger.error(f"Nan-loss at epoch {ep}")
                raise ValueError(f"Nan-Loss")
        
            # write loss to tensorboard
            for k, v in loss.items():
                epoch_str += f"{k}: {round(v, 3)} "
                self.tb_writer.add_scalar(f'Train/{k}', v, epoch)
        
            save_best_ckpt = False
            if do_eval:
                epoch_str += "\n\teval: "
                self.model.eval()
                measures = self.eval_epoch(ep, num_samples=self.num_samples)
                
                if self.update_lr and ep > int(0.40 * self.num_epoch):
                    self.lr_scheduler.step(measures[self.main_metric])
                    
                for k, v in measures.items():
                    assert not math.isnan(v), f"{k} got nan at epoch: {ep}"
                    epoch_str += f"{k}: {round(v, 3)} "
                    self.tb_writer.add_scalar(f'Val/{k}', v, epoch)
                
                if measures[self.main_metric] < best_error:
                    save_best_ckpt = True
                    best_error = measures[self.main_metric]
                    epoch_str += " new best"
                    no_change = 0
                
                no_change += 1
                if no_change >= self.patience:
                    logging.info(
                        f"Stopping after {self.patience} epochs without change")
                    break

            self.logger.info(f"{epoch_str}")
            
            # save current model to checkpoint
            if ep % self.config.TRAIN.ckpt_freq == 0 or save_best_ckpt:
                self.save_model(ep, save_best_ckpt)
                
    def eval(self, do_eval: bool = True, do_best: bool = False) -> None:
        """ Evaluates all checkpoints from the corresponding experiment. The 
        method specific to the trainer is eval_epoch(). 
        
        Inputs:
        -------
        do_eval[boolean]: if True it will run validation, otherwise it will 
        run testing.
        """
        tb_name = 'Val'if do_eval else 'Test'
        self.logger.info(f"Running evaluation on {tb_name}!")
        
        base = self.out.ckpts
        if do_best:
            ckpt_files = ['best.pth']
            base = self.out.best
        elif self.config.BASE_CONFIG.load_ckpt:
            ckpt_files = [self.config.BASE_CONFIG.ckpt_name]
        elif self.config.BASE_CONFIG.load_ckpts_from_path:
            base = self.config.BASE_CONFIG.ckpt_path
            ckpt_files = natsorted(os.listdir(self.out.ckpts))
        else:
            ckpt_files = natsorted(os.listdir(self.out.ckpts))
            logging.info(f"Running checkpoints from dir: {self.out.ckpts}")
            
        assert len(ckpt_files) > 0, f"No checkpoints in dir: {self.out.ckpts}"
        
        best_ade = float('inf')
        pbar = tqdm(total=len(ckpt_files))
        for file in ckpt_files:
            ckpt_file = os.path.join(base, file)
            self.load_model(ckpt_file)
            
            self.logger.info(f"{self.name} running checkpoint: {file}")
            if do_best:
                # TODO: deal with this
                epoch = self.num_epoch
            else:
                epoch = int(file.split('_')[-1].split('.')[0])
            epoch_str = f"Epoch[{epoch}/{self.num_epoch}] "
            
            if do_eval:
                measures = self.eval_epoch(
                    epoch=epoch, num_samples=self.num_samples)
            else:
                measures = self.test_epoch(
                    epoch=epoch, num_samples=self.num_samples)
                        
            for k, v in measures.items():
                assert not math.isnan(v), f"{k} got nan at epoch: {epoch}"
                epoch_str += f"{k}: {round(v, 3)} "
                self.tb_writer.add_scalar(f'{tb_name}/{k}', v, epoch)
            
            if best_ade > measures[self.main_metric]:
                best_ade = measures[self.main_metric]
                epoch_str += " new best"
            
            self.logger.info(f"{epoch_str}")
            pbar.update(1)

    def create_output_dirs(self) -> None:
        """ Creates the experiment name-tag and all output directories. """
        # create the experiment tag name
        exp_name = "{}-{}_EXP-{}DHL{}FL{}PL{}ST{}LR{}WM{}{}".format(
            self.trainer, 
            self.coord, 
            self.dim, 
            self.config.TRAJECTORY.hist_len,  
            self.config.TRAJECTORY.fut_len, 
            self.config.TRAJECTORY.pat_len, 
            self.config.TRAJECTORY.step,
            self.config.TRAIN.lr,
            int(self.config.TRAIN.warmup),
            self.config.BASE_CONFIG.exp_tag
        )

        # create all output directories
        out = os.path.join(
            self.config.BASE_CONFIG.out_dir, self.config.DATASET.name, exp_name)
        if not os.path.exists(out):
            os.makedirs(out)

        # create subdirs required for the experiments
        assert not self.config.BASE_CONFIG.sub_dirs == None, \
            f"No sub-dirs were specified!"

        self.out = {}
        for sub_dir in self.config.BASE_CONFIG.sub_dirs:
            self.out[sub_dir] = os.path.join(out, sub_dir)
            if not os.path.exists(self.out[sub_dir]):
                os.makedirs(self.out[sub_dir])
        self.out = mutils.dotdict(self.out)
        self.out.base = out
        self.config.VISUALIZATION.plot_path = self.out.plots
        self.config.VISUALIZATION.video_path = self.out.videos

        # tensorboard writer
        tb_dir = os.path.join(out, 'tb')
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)
        self.tb_writer = SummaryWriter(tb_dir)
    
        self.logger = logging.getLogger(__name__)
         
        level = (logging.DEBUG 
            if self.config.BASE_CONFIG.log_mode == "debug" else logging.INFO)
        output_log = os.path.join(out, self.config.BASE_CONFIG.log_file)
        logging.basicConfig(
            filename=output_log, filemode='a', level=level, format=FORMAT, 
            datefmt='%Y-%m-%d %H:%M:%S')
        
        self.logger.info(f"{self.name} created output directory: {out}")
        
    def save_config(
        self, config: dict, filename: str = 'config.json', log_dump: bool = True
    ) -> None:
        """ Saves a copy of the configuration file. 
        
        Inputs
        ------
        config[dict]: configuration parameters
        filename[str]: name of fila to save.
        log_dump[bool]: writes the configuration to the self.logger.  
        """
        # saves a copy of the config file used for the current experiment
        json_filename = os.path.join(self.out.base, filename)
        with open(json_filename, 'w') as json_file:
            json.dump(config, json_file, indent=2)
        
        if log_dump:
            self.logger.info(f"Config:\n{json.dumps(config, indent=2)}")
            self.logger.info(f"{self.name} saved the configuration to: {json_filename}")
        
    def save_model(self, epoch: int, save_best: bool) -> None:
        """ Saves a predictor model, optimizer and lr scheduler to specified 
        filename.
        
        Inputs:
        -------
        epoch[int]: epoch number of corresponding model.
        """
        ckpt_file = os.path.join(self.out.ckpts, f'ckpt_{epoch}.pth')
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict()}, ckpt_file)
        self.logger.info(f"{self.name} saved checkpoint to: {ckpt_file}")
        
        if save_best:
            ckpt_file = os.path.join(self.out.best, f'best.pth')
            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict()}, ckpt_file)

    def load_model(self, filename: str) -> None:
        """ Loads a predictor model from specified filename.
        
        Inputs:
        -------
        filename[str]: checkpoint filename. 
        """
        self.logger.debug(f"{self.name} loading checkpoint from: {filename}")
        model = torch.load(filename, map_location=torch.device('cpu'))
        self.model.load_state_dict(model['model'])
        self.optimizer.load_state_dict(model['optimizer'])
        self.lr_scheduler.load_state_dict(model['lr_scheduler'])
    
    def create_visualizations(
        self, hist: torch.tensor, fut: torch.tensor, preds: torch.tensor, 
        best_sample_idx: torch.tensor, seq_start_end: torch.tensor, 
        filename: str, **kwargs) -> None: 
        """ Generates visualizations for ground-truth and predicted trajectories.
        And saves predictions into npy arrays. 
        
        Inputs:
        -------
        hist[torch.tensor]: trajectory observed history
        fut[torch.tensor]: trajectory future
        preds[torch.tensor]: predicted trajectories
        best_sample_idx[torch.tensor]: index of best predicted trajectory based
            on MinADE. 
        seq_start_end[torch.tensor]: tensor indicating where scenes start/end.
        filename[str]: plot filename. 
        """
        # pred = pred_list[best_sample_idx]
        preds = preds.cpu() if preds.is_cuda else preds
        preds = torch.transpose(preds, 2, 3).numpy()
        # np.save(
        #     os.path.join(self.out.trajs, f"traj-{i}_pred.npy"), pred)
    
        hist = hist.cpu() if hist.is_cuda else hist
        hist = torch.transpose(hist, 1, 2).numpy()
        # np.save(
        #     os.path.join(self.out.trajs, f"traj-{i}_hist.npy"), hist)
        
        fut = fut.cpu() if fut.is_cuda else fut
        fut = torch.transpose(fut, 1, 2).numpy()
        # np.save(
        #     os.path.join(self.out.trajs, f"traj-{i}_fut.npy"), fut)
        
        best_sample_idx = (best_sample_idx.cpu() 
            if best_sample_idx.is_cuda else best_sample_idx)
        
        patterns = kwargs.get('patterns')
        if torch.is_tensor(patterns):
            patterns = patterns.cpu() if patterns.is_cuda else patterns
            patterns = patterns.numpy()
        
        vis.plot_trajectories(
            self.config.VISUALIZATION, hist, fut, preds, seq_start_end, 
            best_sample_idx.numpy(), filename, patterns=patterns)

    def compute_loss(self, **kwargs) -> dict:
        """ Computes trainer's loss.
        
        Inputs:
        -------
        **kwargs: keyword arguments as needed by the trainer. 
        
        Outputs:
        --------
        loss[dict]: dictionary containing all computed losses. 
        """
        epoch = kwargs.get('epoch')
        
        kld, kld_item = torch.tensor(0.0), 0.0
        if not kwargs.get('kld') is None:
            kld = kwargs.get('kld')
            kld_item = kld.item()
            
        nll, nll_item = torch.tensor(0.0), 0.0
        if not kwargs.get('nll') is None:
            nll = kwargs.get('nll')
            nll_item = nll.item()
            
        mse, mse_item = torch.tensor(0.0), 0.0
        if not kwargs.get('mse') is None:
            mse = kwargs.get('mse')
            mse_item = mse.item()

        return {
            'Loss': (self.warmup[epoch-1] * kld + nll + mse),
            'LossKLD': kld_item, 
            'LossNLL': nll_item,
            'LossMSE': mse_item,
        }

    # --------------------------------------------------------------------------
    # All methods below should be implemented by trainers that inherit from the
    # BaseTrainer class.
    # --------------------------------------------------------------------------
    def train_epoch(self, epoch: int, **kwargs) -> dict:
        """ Trains one epoch. 
        
        Inputs:
        -------
        epoch[int]: epoch number to test. 
        **kwargs: keyword arguments as needed by the trainer. 
        
        Outputs:
        --------
        loss[dict]: dictionary containing all of losses computed during training. 
        """
        error_msg = f"train_epoch() should be implemented by {self.name}"
        self.logger.error(error_msg)
        raise NotImplementedError(error_msg)

    @torch.no_grad()
    def eval_epoch(self, epoch: int, **kwargs) -> dict:
        """ Evaluates one epoch. 
        
        Inputs:
        -------
        epoch[int]: epoch number to test. 
        **kwargs: keyword arguments as needed by the trainer. 
        
        Outputs:
        --------
        loss[dict]: dictionary containing all of losses computed during training. 
        """
        error_msg = f"eval_epoch() should be implemented by {self.name}"
        self.logger.error(error_msg)
        raise NotImplementedError(error_msg)

    @torch.no_grad()
    def test_epoch(self, epoch: int, **kwargs) -> dict:
        """ Evaluates one epoch. 
        
        Inputs:
        -------
        epoch[int]: epoch number to test. 
        **kwargs: keyword arguments as needed by the trainer. 
        
        Outputs:
        --------
        loss[dict]: dictionary containing all of losses computed during training. 
        """
        error_msg = f"test_epoch() should be implemented by {self.name}"
        self.logger.error(error_msg)
        raise NotImplementedError(error_msg)

    def setup(self) -> None:
        """ Initializes the model, optimizer, lr_scheduler, etc. """
        error_msg = f"setup() should be implemented by {self.name}"
        self.logger.error(error_msg)
        raise NotImplementedError(error_msg)