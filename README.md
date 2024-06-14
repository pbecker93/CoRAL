# "Combining Reconstruction and Contrastive Methods for Multimodal Representations in RL"
**Philipp Becker, Sebastian Mossburger, Fabian Otto, Gerhard Neumann** 

*Published at Reinforcement Learning Conference (RLC) 2024*

## Setup
Tested with Python 3.10 

1. Install PyTorch (e.g. ```pip install torch torchvision torchaudio``` or ```conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia``` (tested with PyTorch 1.13.1 CUDA 11.7 installed over conda)
2. navigate to this folder and run ```pip install -e .``` . it should install all further requirements

### Download Data for Natural Video Backgrounds and Occlusions
For experiments with natural video backgrounds and occlusions you first need to download the corresponding data and 
set the paths in ```envs/distactor_paths.yml```.

**Natural Background (Kinetics400)**
you can use the script ```envs/util/download_kinetics.py``` to download the data. Note that this will requires ```pytube```
(install via pip) and will download several GB of data.

**Occlusions**
Download from https://figshare.com/s/96fb07a704dfb127b6f8 (about 400 MB), unpack and set the path in ```envs/distactor_paths.yml```.

### Setup Maniskill2
The Manipulation tasks build on Maniskill2, please follow their installation instruction https://haosulab.github.io/ManiSkill2/getting_started/installation.html#

Download the ReplicaCAD Background Scenes from here https://huggingface.co/datasets/ai-habitat/ReplicaCAD_baked_lighting 
and add them to the $MS2_ASSET_DIR folder  which should now look like:
```
$MS2_ASSET_DIR
| - partnet_mobility      # Faucet and Drawer models 
| - stages_uncompressed   # ReplicaCAD Scenes
| - <potentially other folders>
```
## Running 

We provide several example configs in ```experiments/configs/*_example.yml```. For example, to run the model-free example run

```python experiments/run_rl.py experiments/configs/model_free_example.yml```

**Logging:** 
Currently, only logging to console is enabled. To enable logging to wandb import the ```if __name__ == "__main__":``` part in
```experiments/run_rl.py```and add the wandb entity to the config. (see ```experiments/configs/model_free_example.yml``` ) 

**Experiments from Paper:** 
The configs used for the results provided in the paper can be found under ```experiments/configs/model_free``` and ```experiments/configs/model_based```.
For example, to run train the model-free agents for standard images with Joint(CV) run 

```python experiments/run_rl.py experiments/configs/model_free/standard_joint.yml -e mf_standard_joint_cv```

The respective ```defaults``` folders contain the default configs for the individual methods   
