# ------------------------------------------------------------------------------
# @file:    trajair_dataset.py
# @brief:   Contains the TrajAir DataLoader.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# general includes
import logging
import os
import math
import numpy as np
import torch

from torch.utils.data import Dataset

from sprnn.utils.common import poly_fit

logger = logging.getLogger(__name__)

class SDDDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, pat_len=8, step=1, skip=1, 
        min_agent=2, max_agent=21, threshold=0.002, process = 1.0, delim=' '):
        """ Dataset loader for the TrajAir dataset. 
        
        Inputs:
        -------
            data_dir[str]: directory containing dataset files in the format
                <frame_id> <agent_id> <x> <y> <z> <wind_x> <wind_y>
            obs_len[int]: number of time-steps in input trajectories
            pred_len[int]: number of time-steps in output trajectories
            pat_len[int]: number of time-steps in pattern
            step[int]: subsampling step for the observed trajectory
            skip[int]: number of consecutive frames to skip while making the dataset
            min_agen[int]: minimum number of agentes that should be in a seqeunce 
            max_agen[int]: maximum number of agentes that should be in a seqeunce 
            process[float]: percentage of the data to process
            delim[str]: delimiter in the dataset files
        """
        super(SDDDataset, self).__init__()
        
        dim = 2
        
        obs_final_len = int(math.ceil(obs_len / step))
        pred_final_len = int(math.ceil(pred_len / step))
        
        seq_len = obs_len + pred_len
        seq_final_len = obs_final_len + pred_final_len
        
        files = np.load(data_dir+"/trajectories.npy", allow_pickle=True)
        
        num_peds_in_seq = []
        loss_mask_list = []
        non_linear_ped = []
        seq_list = []
        seq_list_rel = []
        pat_seq_list = []
        pat_seq_list_rel = []
        
        for file in files:
            sequences_in_curr_file = []

            scene_name = file[0]
            data = file[1]
        
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                # List of list: contains all positions in each frame
                frame_data.append(data[frame == data[:, 0], :])
                
            # Num sequences to consider for a specified sequence length
            # (seq_len = obs_len + pred_len)
            num_sequences = int(math.ceil((len(frames) - seq_len + 1) / skip))
            
            for idx in range(0, num_sequences * skip + 1, skip):
                # All data for the current sequence: from the current index to 
                # current_index + sequence length
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + seq_len], axis=0)

                # IDs of pedestrians in the current sequence
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                num_peds = len(peds_in_curr_seq)
                # if num_peds < min_agent or num_peds > max_agent:
                #     continue
                
                curr_seq_rel = np.zeros((num_peds, dim, seq_len))
                curr_seq = np.zeros((num_peds, dim, seq_len))
                
                pat_seq = np.zeros((num_peds, obs_final_len+1, dim, pat_len))
                pat_seq_rel = np.zeros((num_peds, obs_final_len+1, dim, pat_len-1))
                
                curr_loss_mask = np.zeros((num_peds, seq_len))
                
                num_peds_considered = 0
                _non_linear_ped = []
                
                # Cycle on pedestrians for the current sequence index
                for _, ped_id in enumerate(peds_in_curr_seq):
                    # import pdb; pdb.set_trace()
                    # Current sequence for each pedestrian
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)

                    # Start frame for the current sequence of the current 
                    # pedestrian reported to 0
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    
                    # End frame for the current sequence of the current pedestrian:
                    # end of current pedestrian path in the current sequence.
                    # It can be sequence length if the pedestrian appears in all
                    # frame of the sequence or less if it disappears earlier.
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

                    # Exclude trajectories less then seq_len
                    if pad_end - pad_front != seq_len:
                        continue
            
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])

                    # To avoid indexing issues, compute one pattern per time-step
                    pattern_seq = curr_ped_seq[:3]
                    patterns_rel = []
                    patterns = []
                    for t in range(obs_final_len+1):
                        pat = pattern_seq[:, t:t+pat_len]
                        pat_rel = pat[:, 1:] - pat[:, :-1]
                        # import matplotlib.pyplot as plt
                        # plt.plot(pat[0, :], pat[1, :], c='b', lw=3)
                        # plt.plot(agent_seq[0, :], agent_seq[1, :], c='r')
                        # plt.show()
                        # plt.close()
                        patterns.append(pat)
                        patterns_rel.append(pat_rel)
                                            
                    patterns = np.stack(patterns)
                    patterns_rel = np.stack(patterns_rel)
                    
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]

                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    
                    pat_seq[_idx] = patterns
                    pat_seq_rel[_idx] = patterns_rel
                    
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_agent:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    sequences_in_curr_file.append(curr_seq[:num_peds_considered])
                    
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    
                    pat_seq_list.append(pat_seq[:num_peds_considered])
                    pat_seq_list_rel.append(pat_seq_rel[:num_peds_considered])

            if len(sequences_in_curr_file) > 0:
                all_traj = np.concatenate(sequences_in_curr_file, axis=0)

        self.num_seq = len(seq_list)
        self.seq_list = np.concatenate(seq_list, axis=0)
        self.seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        self.pat_seq_list = np.concatenate(pat_seq_list, axis=0)
        self.pat_seq_list_rel = np.concatenate(pat_seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
                
        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(self.seq_list[:, :, :obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(self.seq_list[:, :, obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(self.seq_list_rel[:, :, :obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(self.seq_list_rel[:, :, obs_len:]).type(torch.float)
        self.obs_patt = torch.from_numpy(self.pat_seq_list).type(torch.float)
        self.obs_pat_rel = torch.from_numpy(self.pat_seq_list_rel).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        
        # collect idxs to identificate the start-end indices (inside the batch 
        # dim) for single sequences
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :],
            self.pred_traj[start:end, :],
            self.obs_patt[start:end, :],
            self.obs_traj_rel[start:end, :],
            self.pred_traj_rel[start:end, :],
            self.obs_pat_rel[start:end, :],
            # self.non_linear_ped[start:end, :], 
            # self.loss_mask[start:end, :]
        ]
        return out

def sdd_seq_collate(data):
    (
        obs_traj, 
        pred_traj, 
        pat_traj,
        obs_traj_rel, 
        pred_traj_rel,
        pat_traj_rel, 
        # non_linear_ped_list, 
        # loss_mask_list
    ) = zip(*data)
    
    _len = [len(seq) for seq in obs_traj]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # pat_len = pat_len[0]
    batch_size = len(obs_traj)
    obs_seq_len, n_agents, features = obs_traj[0].shape
    pred_seq_len, _, _ = pred_traj[0].shape

    # nbatch, dim, seq_len -> seq_len, batch, dim
    obs_traj = torch.cat(obs_traj, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_traj_rel, dim=0).permute(2, 0, 1)
    
    pred_traj = torch.cat(pred_traj, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_traj_rel, dim=0).permute(2, 0, 1)
    
    # nbatch, seq_len, dim, pat_len -> seq_len, nbatch, dim, pat_len
    pat_traj = torch.cat(pat_traj, dim=0).permute(1, 0, 3, 2)
    pat_traj_rel = torch.cat(pat_traj_rel, dim=0).permute(1, 0, 3, 2)

    seq_start_end = torch.LongTensor(seq_start_end)
    
    # TODO: 
    context = torch.empty([])
    
    out = [
        obs_traj, pat_traj, pred_traj, 
        obs_traj_rel, pat_traj_rel, pred_traj_rel, 
        context, seq_start_end
    ]
    return tuple(out)