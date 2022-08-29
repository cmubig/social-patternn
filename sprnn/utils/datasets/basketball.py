# ------------------------------------------------------------------------------
# @file:    bsk.py
# @brief:   Contains utility functions to preprocess the BSK dataset.
# ------------------------------------------------------------------------------
import logging
import numpy as np
import torch

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BasketballDataset(Dataset):
    """ Dataloader for the NBA dataset. """
    def __init__(
        self, path, n_agents, obs_len=10, pred_len=40, pat_len=5, step=1, 
        process=1.0):
        """ Dataset loader for the TrajAir dataset. 
        
        Inputs:
        -------
            path[str]: directory containing dataset files in the format
            obs_len[int]: number of time-steps in input trajectories
            pred_len[int]: number of time-steps in output trajectories
            pat_len[int]: number of time-steps in pattern
            step[int]: subsampling step for the observed trajectory
            process[float]: percentage of the data to process
        """
        super(BasketballDataset, self).__init__()

        self.data_dir = path
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.n_agents = n_agents + 1 # +1 is the ball 
        self.pat_len = pat_len
        self.step = step
        
        traj_abs = np.load(self.data_dir+"/trajectories.npy")
        assert traj_abs.shape[0] == self.seq_len
    
        num_seqs = traj_abs.shape[1] // self.n_agents
        idxs = [idx for idx in range(0, (num_seqs * self.n_agents) + self.n_agents, self.n_agents)]
        seq_start_end = [[start, end] for start, end in zip(idxs[:], idxs[1:])]

        self.num_samples = len(seq_start_end)

        traj_rel = np.zeros(traj_abs.shape)
        traj_rel[1:, :, :] = traj_abs[1:, :, :] - traj_abs[:-1, :, :]

        self.obs_traj = torch.from_numpy(traj_abs).type(torch.float)[:obs_len, :, :]        
        self.obs_traj_rel = torch.from_numpy(traj_rel).type(torch.float)[:obs_len, :, :]
        
        self.pat_traj = torch.from_numpy(traj_abs).type(torch.float)[:, :, :]
        
        self.pred_traj = torch.from_numpy(traj_abs).type(torch.float)[obs_len:, :, :]
        self.pred_traj_rel = torch.from_numpy(traj_rel).type(torch.float)[obs_len:, :, :]
        
        self.seq_start_end = seq_start_end

    def __len__(self):
        return self.num_samples

    def __max_agents__(self):
        # in bball the number of agents per scene is always the same
        return self.n_agents

    def __getitem__(self, idx):
        start, end = self.seq_start_end[idx]
        out = [
            self.obs_traj[:, start:end, :], 
            self.pred_traj[:, start:end, :],
            self.pat_traj[:, start:end, :], 
            self.obs_traj_rel[:, start:end, :], 
            self.pred_traj_rel[:, start:end, :],
            self.pat_len
        ]
        return out

def basketball_seq_collate(data):
    (
        obs_traj, 
        pred_traj, 
        pat_traj,
        obs_traj_rel, 
        pred_traj_rel,
        pat_len
    ) = zip(*data)
    
    pat_len = pat_len[0]
    batch_size = len(obs_traj)
    obs_seq_len, n_agents, features = obs_traj[0].shape
    pred_seq_len, _, _ = pred_traj[0].shape

    obs_traj = torch.cat(obs_traj, dim=1)
    pred_traj = torch.cat(pred_traj, dim=1)

    obs_traj_rel = torch.cat(obs_traj_rel, dim=1)
    
    pred_traj_rel = torch.cat(pred_traj_rel, dim=1)

    pat_traj = torch.cat(pat_traj, dim=1)
    patterns = []
    patterns_rel = []
    for t in range(obs_seq_len+1):
        pat = pat_traj[t:t+pat_len]
        pat_vec = pat[1:] - pat[:-1]
        patterns.append(pat)
        patterns_rel.append(pat_vec)
    
    # seq_len, pat_len, agents, dim -> seq_len, agents, dim, pattern_len
    patterns = torch.stack(patterns).permute(0, 2, 1, 3)
    patterns_rel = torch.stack(patterns_rel).permute(0, 2, 1, 3)
    
    # fixed number of agent for every play -> we can manually build seq_start_end
    idxs = list(range(0, (batch_size * n_agents) + n_agents, n_agents))
    seq_start_end = [[start, end] for start, end in zip(idxs[:], idxs[1:])]
    seq_start_end = torch.LongTensor(seq_start_end)
    
    # 1 - ball, 2 - atk, 3 - def
    context = torch.tensor(
        [[1,0,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0], 
         [0,0,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1]])
    context = context.repeat(batch_size, 1)
    
    # patterns 
    out = [
        obs_traj, patterns, pred_traj, 
        obs_traj_rel, patterns_rel, pred_traj_rel, 
        context, seq_start_end
    ]
    return tuple(out)