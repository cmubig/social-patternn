# ------------------------------------------------------------------------------
# @file:    common.py
# @brief:   This file contains the implementation common utility classes and 
#           function needed by the modules in sprnn.
# ------------------------------------------------------------------------------
from numpy import dtype, mask_indices
import torch
import numpy as np

# definitions below
DIMS = [2, 3, 6]
COORDS = ["rel", "abs"]
TRAJ_ENCODING_TYPE = ["mlp", "tcn"]
ADJ_TYPE = ["fc", "fully-connected", "distance-similarity", "knn", "gaze"]
FORMAT = '[%(asctime)s: %(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'

# classes below:

class Config:
    """ A class for holding configuration parameters. """
    def __init__(self, config):
        self.BASE_CONFIG = dotdict(config)
        
        self.GOAL = None
        if self.BASE_CONFIG.goal:
            self.GOAL = dotdict(self.BASE_CONFIG.goal)
            
        self.TRAJECTORY = None
        if self.BASE_CONFIG.trajectory:
            self.TRAJECTORY = dotdict(self.BASE_CONFIG.trajectory)

        self.DATASET = None
        if self.BASE_CONFIG.dataset:
            self.DATASET = dotdict(self.BASE_CONFIG.dataset)

        self.TRAIN = None
        if self.BASE_CONFIG.training_details:
            self.TRAIN = dotdict(self.BASE_CONFIG.training_details)

        self.MODEL = None
        if self.BASE_CONFIG.model_design:
            self.MODEL = dotdict(self.BASE_CONFIG.model_design)

        self.VISUALIZATION = None
        if self.BASE_CONFIG.visualization:
            self.VISUALIZATION = dotdict(self.BASE_CONFIG.visualization)
            
        self.MAP = None
        if self.BASE_CONFIG.map:
            self.MAP = dotdict(self.BASE_CONFIG.map)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# methods below:

def convert_rel_to_abs(traj_rel, start_pos, permute: bool = False):
    """ Converts a trajectory expressed in relative displacement to an absolute
    values given a start position. 
    
    Inputs:
    -------
    traj_rel[torch.tensor(batch, seq_len, dim)]: trajectory of displacements
    start_pos[torch.tensor(batch, dim)]: initial absolute position
   
    Outputs:
    --------
    traj_abs[torch.tensor(seq_len, batch, 2)]: trajectory of absolute coords
    """
    if permute:
        # (seq_len, batch, 2) -> (batch, seq_len, 2)
        traj_rel = traj_rel.permute(1, 0, 2)        
        
    displacement = torch.cumsum(traj_rel, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    
    if permute:
        return abs_traj.permute(1, 0, 2)
    return abs_traj

def compute_social_influences(
    traj: torch.tensor, goals: torch.tensor, seq_start_end: torch.tensor, 
    max_agents: int, flatten: bool = True,
) -> torch.tensor:
    """ Computes the agent-to-agents displacements from a position x_t in a 
    trajectory to the sub-goals in a pattern trajectory p_t of all agents. 
    
    Inputs:
    -------
    traj[torch.tensor]: a trajectory segment for which distances w.r.t the
        patterns will be computed. 
    patterns[torch.tensor]: a pattern trajectory it has to be at least longer  
        than the trajectory length + subgoal length + dt.
    pat_len[int]: length of the pattern subgoal which will be used to compute
        distances at time t. 
    max_agents[int]: K nearest neighbors to compute displacement.
    flatten[bool]: if True, will flatten the final social influence matrix. 

    Outputs:
    --------
    displacements[torch.tensor]: displacements-to-subgoal vector.
    """
    traj_len, batch_size, dim = traj.shape
    displacements = torch.zeros((traj_len, batch_size, max_agents, dim))

    for start, end in seq_start_end:
        num_agents = end - start
        
        for i in range(start, end):
            disp = goals[:traj_len, start:end] - traj[:traj_len, i].unsqueeze(1)
            
            # TODO: make this faster
            if num_agents > max_agents:
                dist = torch.sqrt(torch.sum(disp ** 2, axis=2))
                _, idx = torch.topk(dist, max_agents, largest=False)
                
                for t in range(traj_len):
                    displacements[t, i, :] = disp[t, idx[t]]
            else: 
                displacements[:, i, :num_agents] = disp[:, :num_agents]
    
    if flatten:
        return displacements.flatten(start_dim=2)
       
    return displacements

def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            if "?" in line:
                continue
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0