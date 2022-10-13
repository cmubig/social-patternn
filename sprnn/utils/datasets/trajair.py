# ------------------------------------------------------------------------------
# @file:    trajair_dataset.py
# @brief:   Contains the TrajAir DataLoader for top-down view trajectory.
# ------------------------------------------------------------------------------

# general includes
import logging
import os
import math
import numpy as np
import torch

from torch.utils.data import Dataset

from sprnn.utils.common import read_file

logger = logging.getLogger(__name__)

class TrajAirDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=40, pred_len=100, pat_len=7, step=5, 
        skip=2, min_agent=1, max_agent=10, process=1.0, delim=' '):
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
        super(TrajAirDataset, self).__init__()
        
        dim = 3
        
        obs_final_len = int(math.ceil(obs_len / step))
        pred_final_len = int(math.ceil(pred_len / step))
        
        seq_len = obs_len + pred_len
        seq_final_len = obs_final_len + pred_final_len
        
        all_files = os.listdir(data_dir)
        all_files = [os.path.join(data_dir, _path) for _path in all_files]
        
        num_num_agents = []
        seq_list = []
        seq_list_rel = []
        pattern_context_list = []
        pattern_context_list_rel = []
        weather_context_list = []

        N = len(all_files)
        for i, path in enumerate(all_files):
            # check if file is empty
            if os.stat(path).st_size == 0:
                continue
            
            prog = (i+1) / N
            if prog > process:
                break

            print(f"[{i+1}/{N}-{round(prog * 100, 3)}%] path: {path}", end="\r")

            data = read_file(path, delim)
            data = np.around(data, decimals=3)

            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])

            num_sequences = int(
                math.ceil((len(frames) - (seq_len) + 1) / skip))
            if num_sequences < 1:
                continue
            
            for idx in range(0, num_sequences * skip + 1, skip):
                seq_data = np.concatenate(frame_data[idx:idx + seq_len], axis=0)
                
                num_agents = np.unique(seq_data[:, 1])
                if len(num_agents) < min_agent or len(num_agents) > max_agent:
                    continue
        
                # data placeholders
                seq = np.zeros((len(num_agents), 3, seq_final_len))
                seq_rel = np.zeros((len(num_agents), 3, seq_final_len))
                
                pat_seq = np.zeros((len(num_agents), obs_final_len+1, dim, pat_len))
                pat_seq_rel = np.zeros((len(num_agents), obs_final_len+1, dim, pat_len-1))
                
                weather = np.zeros((len(num_agents), 2, seq_final_len))
                
                num_agents_considered = 0
                for _, agent_id in enumerate(num_agents):
                    
                    agent_seq = seq_data[seq_data[:, 1] == agent_id, :]
                    pad_front = frames.index(agent_seq[0, 0]) - idx
                    pad_end = frames.index(agent_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != seq_len:
                        continue
                    
                    if agent_seq.shape[0] < seq_len: # isn't this the same as above?
                        continue
    
                    agent_seq = np.transpose(agent_seq[:, 2:])
                    
                    agent_seq_step = agent_seq[:, ::step]
                    obs = agent_seq_step[:, :obs_final_len]
                    pred = agent_seq_step[:, obs_final_len:]
    
                    # create patterns
                    # TODO: interpolate patterns
                    pattern_seq = agent_seq_step[:3]
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
                    
                    agent_seq = np.hstack((obs, pred))[:, :seq_final_len]
                    
                    agent_weather = agent_seq[-2:, :]

                    rel_agent_seq = np.zeros(agent_seq.shape)
                    rel_agent_seq[:, 1:] = agent_seq[:, 1:] - agent_seq[:, :-1]
                    
                    _idx = num_agents_considered
                    if (agent_seq.shape[1] != seq_final_len):
                        continue
                 
                    if pad_front != 0:
                        raise ValueError(idx, pad_front)
                    
                    seq[_idx, :, pad_front:pad_end] = agent_seq[:3, :]
                    seq_rel[_idx, :, pad_front:pad_end] = rel_agent_seq[:3, :]
                    pat_seq_rel[_idx, pad_front:pad_end] = patterns_rel
                    pat_seq[_idx, pad_front:pad_end] = patterns
                    weather[_idx, :, pad_front:pad_end] = agent_weather
                    num_agents_considered += 1

                if num_agents_considered >= min_agent:
                    num_num_agents.append(num_agents_considered)
                    seq_list.append(seq[:num_agents_considered])
                    seq_list_rel.append(seq_rel[:num_agents_considered])
                    pattern_context_list.append(pat_seq[:num_agents_considered])
                    pattern_context_list_rel.append(pat_seq_rel[:num_agents_considered])
                    weather_context_list.append(weather[:num_agents_considered])
                    
        print("")
        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        pattern_context_list = np.concatenate(pattern_context_list, axis=0)
        pattern_context_list_rel = np.concatenate(pattern_context_list_rel, axis=0)
        weather_context_list = np.concatenate(weather_context_list, axis=0)
        
        # Convert numpy -> Torch Tensor
        # absolute coordinates
        self.hist = torch.from_numpy(
            seq_list[:, :, :obs_final_len]).type(torch.float)
        self.fut = torch.from_numpy(
            seq_list[:, :, obs_final_len:]).type(torch.float)
        self.weather_context = torch.from_numpy(
            weather_context_list[:, :, :]).type(torch.float)
        self.pattern_context = torch.from_numpy(
            pattern_context_list).type(torch.float)
        
        # relative
        self.hist_rel = torch.from_numpy(
            seq_list_rel[:, :, :obs_final_len]).type(torch.float)
        self.fut_rel = torch.from_numpy(
            seq_list_rel[:, :, obs_final_len:]).type(torch.float)
        self.pattern_context_rel = torch.from_numpy(
            pattern_context_list_rel).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_num_agents).tolist()
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        
        self.max_agents = -float('Inf')
        for (start, end) in self.seq_start_end:
            n_agents = end - start
            self.max_agents = n_agents if n_agents > self.max_agents else self.max_agents

    def __len__(self):
        return self.num_seq

    def __max_agents__(self):
        return self.max_agents

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.hist[start:end, :],
            self.fut[start:end, :],
            self.pattern_context[start:end, :],
            self.hist_rel[start:end, :],
            self.fut_rel[start:end, :],
            self.pattern_context_rel[start:end, :],
            self.weather_context[start:end, :],
        ]
        return out

def trajair_seq_collate(data):
    (
        hist_list,
        fut_list,
        pat_context_list,
        hist_rel_list,
        fut_rel_list,
        pat_rel_context_list,
        weather_context_list,
    ) = zip(*data)

    _len = [len(seq) for seq in hist_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(hist_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(fut_list, dim=0).permute(2, 0, 1)
    obs_pat_seq = torch.cat(pat_context_list, dim=0).permute(1, 0, 3, 2)#.permute(2, 0, 1)
   
    # seq_len, agents, dim, pattern_len
    obs_traj_rel = torch.cat(hist_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(fut_rel_list, dim=0).permute(2, 0, 1)
    obs_pat_seq_rel = torch.cat(pat_rel_context_list, dim=0).permute(1, 0, 3, 2)#.permute(2, 0, 1)
    weather_context = torch.cat(weather_context_list, dim=0).permute(2, 0, 1)
    seq_start_end = torch.LongTensor(seq_start_end)

    out = [
        obs_traj, obs_pat_seq, pred_traj, 
        obs_traj_rel, obs_pat_seq_rel, pred_traj_rel, 
        weather_context, seq_start_end
    ]
    return tuple(out)