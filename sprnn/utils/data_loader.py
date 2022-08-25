# ------------------------------------------------------------------------------
# @file:    data_loader.py
# @brief:   Contains utility functions to preprocess the datasets.
# ------------------------------------------------------------------------------
import logging
import os
import numpy as np

from torch.utils.data import DataLoader

from sprnn.utils.datasets.trajair import TrajAirDataset, trajair_seq_collate
from sprnn.utils.datasets.sdd import SDDDataset, sdd_seq_collate
from sprnn.utils.datasets.basketball import BasketballDataset, basketball_seq_collate

logger = logging.getLogger(__name__)

def load_data(data_config: dict, traj_config: dict):
    """ Loads the data for training, validating and testing depending on the
    configuration. 
    Inputs:
        config: Contains all configuration parameters needed to load the data
    Outputs:
        data loaders for training, validation and testing.
    """
    txt_path = data_config.txt_path
    npy_path = data_config.npy_path
    name = data_config.name
    loader_type = data_config.loader_type

    # Name-tag the experiment
    hl, fl, pl = traj_config.hist_len, traj_config.fut_len, traj_config.pat_len
    st, sk = traj_config.step, traj_config.skip
    mn, mx = traj_config.min_agents, traj_config.max_agents
    out_name = f"{name}-{loader_type}_TAG-HL{hl}FL{fl}PL{pl}TS{st}SK{sk}MN{mn}MX{mx}.npy"
    
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)
    npy_file = os.path.join(npy_path, out_name)
    
    if data_config.load_npy:
        if os.path.exists(npy_file):
            logger.info(f"Loading data from {npy_file}...")
            train_loader, val_loader, test_loader = np.load(
                npy_file, allow_pickle=True)
            return train_loader, val_loader, test_loader
        logger.info(f"{npy_file} does not exist. Preprocessing data instead.")

    in_path = os.path.join(txt_path, name)
    assert os.path.exists(in_path), f"Path {in_path} does not exist!"
    logger.info(f"Loading data from {in_path}")

    # Prepare datasets
    loaders = []
    folders = ["/train", "/val", "/test"]
    shuffle = [True, True, False]

    if loader_type == "trajair":
        
        for i, f in enumerate(folders):
            logger.info(f"Processing data in {f}...")
            data = TrajAirDataset(
                in_path + f, obs_len=hl, pred_len=fl, pat_len=pl, step=st,
                skip=sk, min_agent=mn, max_agent=mx, process=data_config.process)
            
            loader = DataLoader(
                data, 
                batch_size=data_config.train_batch_size,
                num_workers=data_config.loader_num_workers,
                shuffle=shuffle[i],
                collate_fn=trajair_seq_collate
            )
            logger.info(f"...processed!")
            loaders.append(loader)
            
    elif loader_type == "sdd":
        
        for i, f in enumerate(folders):
            logger.info(f"Processing data in {f}...")
            data = SDDDataset(
                in_path + f, obs_len=hl, pred_len=fl, pat_len=pl, step=st,
                skip=sk, min_agent=mn, max_agent=mx, process=data_config.process)
        
            loader = DataLoader(
                data, 
                batch_size=data_config.train_batch_size,
                num_workers=data_config.loader_num_workers,
                shuffle=shuffle[i],
                collate_fn=sdd_seq_collate
            )
            logger.info(f"...processed!")
            loaders.append(loader)
            
    elif loader_type == "bsk":
        
        for i, f in enumerate(folders):
            logger.info(f"Processing data in {f}...")
            data = BasketballDataset(
                in_path + f, n_agents=mx, obs_len=hl, pred_len=fl, pat_len=pl, 
                step=st, process=data_config.process)
        
            loader = DataLoader(
                data, 
                batch_size=data_config.train_batch_size,
                num_workers=data_config.loader_num_workers,
                shuffle=shuffle[i],
                collate_fn=basketball_seq_collate
            )
            logger.info(f"...processed!")
            loaders.append(loader)
            
    else:
        raise NotImplementedError(f"Loader type {loader_type} not implemented")

    logger.info(f"Saving data to {npy_file}")
    np.save(npy_file, loaders)

    logger.info(f"Done!")
    return loaders