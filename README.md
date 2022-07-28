# Social-PatteRNN

This repository contains the code for the paper:

<h4> 
Social-PatteRNN: Socially-Aware Trajectory Prediction Guided by Motion Patterns
</h4>

Ingrid Navarro and [Jean Oh](https://www.cs.cmu.edu/~./jeanoh/) 

<p align="center">
  <img width="775" height="360" src="./readme/trajpred.png" alt="SocialPatteRNN">
</p>

## Model Overview

Social-PatteRNN is an algorithm for recurrent, multi-modal trajectory prediction 
in multi-agent settings. Our approach guides long-term trajectory prediction by 
learning to predict short-term motion patterns. It then extracts sub-goal 
information from the patterns and aggregates it as social context.
<p align="center">
  <img width="800" src="./readme/model.png" alt="SocialPatteRNN">
</p>

## Installation

Setup a conda environment:
```
conda create --name sprnn python=3.7
conda activate sprnn
```

Download the repository and install requirements:
```
git clone --branch sprnn git@github.com:cmubig/social-patternn.git
cd social-patternn
pip install -e . 
```

## Dataset

We have tested our algorithm on three different datasets:
- TrajAir (111days)
- Stanford Drone Dataset (SDD)
- NBA Dataset (BSK)

We provide instructions and the dataloaders to setup the data 
[here](https://github.com/cmubig/social-patternn/tree/sprnn/data).

## Running the code

This repository provides four baselines:
- ```VRNN```: a Recurrent C-VAE for trajectory prediction
- ```VRNN-PAT```: a VRNN with a context module for pattern learning 
- ```VRNN-SOC-PAT```: a VRNN with a context module for pattern learning and interaction encoding 
- ```Social-PatteRNN```: the full model; a VRNN with a context module for pattern learning and interaction encoding with multi-head attention

All of the parameters related to the trajectory specifications, training 
details and model architectures are provided in the configuration files of each 
baseline and experiment. These configuration files can be found in 
```social-patternn/config/dataset-name```.

The ```run.py``` script controls the training, validation and testing for all 
experiments and datasets. An experiment is specified with the flag ```--exp-config```, 
and the type of process is specified with the flag ```--run-type```:
```
python run.py --exp-config path/to/exp-config.json --run-type [trainval | train | eval | test]
```

#### Running the SocialPatteRNN model

For example, to train and validate the SocialPatteRNN model on the 111days dataset, execute:
```
python run.py --exp-config config/111-days/sprnn.json --run-type trainval
```

To test a trained model, run:
```
python run.py --exp-config config/111-days/sprnn_pat-10.json --run-type test
```

To test or evaluate one of a specific checkpoint, you can specify the checkpoint 
number ```ckpt_num``` if the checkpoint is in the default path or the checkpoint 
path ```ckpt_path``` if not. 

Example with checkpoint number which would load ```ckpt_10.pth```:
```
python run.py --exp-config config/111-days/sprnn.json --run-type test --ckpt_num 10
``` 

Example with checkpoint path number which would load any checkpoints in the given path ```best_ckpts```:
```
python run.py --exp-config config/111-days/sprnn.json --run-type test --ckpt_path best_ckpts/
``` 

For each experiment, we provide the configuration files for all the ablations 
performed in our paper. They are organized as follows:
```graphql
config/
├─ 111-days
|   ├─ base_config.json        
|   ├─ vrnn.json
|   ├─ vrnn_pat.json
|   ├─ vrnn_soc_pat.json
|   ├─ sprnn.json
|   ├─ ...
├─ sdd
|   ├─ ...
├─ bsk_all
|   ├─ ...
```

### Running the motion primitive classifier test

Run the default test as:
```
python sprnn/tests/run_mp.py
``` 

You will need to have a folder named ```data_mp``` within the root folder. 
You can download it from [here](https://drive.google.com/drive/folders/1OP7Un55Ks0GFT_Olux27-Az-Tx_c8DY1?usp=sharing).

## Results

Here we report the performance of our Social-PatteRNN model as well as its
ablations in terms of Average Displacement Error (ADE) and Final Displacement 
Error (FDE).

### TrajAir 111days
| Baselines  | ADE  | FDE  | 
|:----------:|:----:|:----:|
| VRNN       | 0.596 | 1.322 | 
| VRNN + PAT-10| 0.580 | 1.258 | 
| VRNN + SOC + PAT-10   | 0.626 | 1.356 | 
| SPRNN   | **0.557** | **1.184** | 

### SDD 
| Baselines  | ADE  | FDE  | 
|:----------:|:----:|:----:|
| VRNN       | 0.643 | 1.259 | 
| VRNN + PAT-10| 0.562 | 1.149 | 
| VRNN + SOC + PAT-10   | **0.559** | 1.134 | 
| SPRNN   | 0.561 | **1.124** | 

### NBA
| Baselines  | ADE  | FDE  | 
|:----------:|:----:|:----:|
| VRNN       | 9.176 | 14.271 | 
| VRNN + PAT-10| 8.659 | 13.748 | 
| VRNN + SOC + PAT-10   | 8.720 | 12.969 | 
| SPRNN   | **8.104** | **12.051** | 

### TODO: add best checkpoints to repo

## Citing

#### TODO: update this
```tex
@inproceedings{name,
  title={Paper},
  author={Author1 and Author2},
  booktitle={Conference},
  year={2022}
 }
```