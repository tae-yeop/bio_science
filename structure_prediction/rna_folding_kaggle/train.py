from argparse import ArgumentParser
import yaml
import os
from models.rnanet import RNANet
from models.rnanet.config import Config
from pytorch_lightning.loggers import WandbLogger
import wandb

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    config = Config(**yaml_config)

    # wandb
    wandb.login(key=config.wandb_key, host=config.wandb_host)

    wandb_logger = WandbLogger(
            project=config.wandb_project_name, 
            name=config.wandb_run_name + '-' + os.environ.get('SLURM_JOBID', ''), 
            config=config)
    
    # data
    if os.path.exists(os.path.join(config.path_processed, 'data.pt')):
        return ...
    



    # Model
    model = RNANet(config)
    model.train()