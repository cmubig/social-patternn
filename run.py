# ------------------------------------------------------------------------------
# @file:    run.py
# @brief:   This script is used for running trajectory prediction trainers. 
#           Example usage: 
#               python run.py \
#                   --exp path/to/config.json \
#                   --run [ trainval | train | eval | test ]
# ------------------------------------------------------------------------------
import argparse
import json
import os

def run(
    exp: str, run: str, ckpt_num: int, ckpt_path: str, resume: bool, best: bool,
    visualize: bool) -> None:
    assert os.path.exists(exp), f"File {exp} does not exist!"

    # load the configuration files
    exp_file = open(exp)
    exp = json.load(exp_file)

    config_file = open(exp["base_config"])
    config = json.load(config_file)
    config.update(exp)
    
    config["log_file"] = f"{run}.log"
    trainer_type = config['trainer']
    
    if ckpt_num:
        config['load_ckpt'] = True
        config['ckpt_name'] = f"ckpt_{ckpt_num}.pth"
    
    if ckpt_path:
        config['load_ckpts_from_path'] = True
        config['ckpt_path'] = ckpt_path
        
    config['visualization']['visualize'] = visualize

    # choose trainer 
    if trainer_type == "vrnn":
        from sprnn.trajpred_trainers.vrnn import VRNNTrainer as Trainer
    elif trainer_type == "socvrnn":
        from sprnn.trajpred_trainers.socvrnn import SocialVRNNTrainer as Trainer
    elif trainer_type == "patternn":
        from sprnn.trajpred_trainers.patternn import PatteRNNTrainer as Trainer
    elif trainer_type == "socpatternn-mlp" or trainer_type == "socpatternn-mha":
        from sprnn.trajpred_trainers.socpatternn import (
            SocialPatteRNNTrainer as Trainer)
    else:
        raise NotImplementedError(f"Trainer {trainer_type} not supported!")
    
    trainer = Trainer(config=config)
    
    if run == "trainval":
        trainer.train(do_eval=True, resume=resume)
    elif run == "train":
        trainer.train(resume=resume)
    elif run == "eval":
        trainer.eval(do_best=best)
    elif run == "test":
        trainer.eval(do_eval=False, do_best=best)
    
    # this is just to have a copy of the original config
    config['load_ckpt'] = False
    config['load_ckpts_from_path'] = False
    trainer.save_config(
        config, filename=f'config_{run}.json', log_dump=False)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp', 
        default='./config/test.json', 
        type=str,
        help='path to experiment configuration file')
    parser.add_argument(
        '--run', 
        default='trainval', 
        type=str, 
        choices=['trainval', 'train', 'eval', 'test'], 
        help='type of experiment [trainval | train | eval | test]')
    parser.add_argument(
        '--ckpt-num', 
        required=False, 
        type=int, 
        help='checkpoint number to evaluate')
    parser.add_argument(
        '--ckpt-path', 
        required=False, 
        type=str, 
        help='path to checkpoint to run')
    parser.add_argument(
        '--resume', 
        required=False, 
        action='store_true', 
        help='resume training process')
    parser.add_argument(
        '--best', 
        required=False, 
        action='store_true', 
        help='enable visualizations')
    parser.add_argument(
        '--visualize', 
        required=False, 
        action='store_true', 
        help='enable visualizations')
    args = parser.parse_args()
    
    run(**vars(args))
    
if __name__ == "__main__":
    main()