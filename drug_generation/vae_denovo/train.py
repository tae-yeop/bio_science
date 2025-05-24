import torch
from torch.utils.data import DataLoader
from dataset import StringDataset
# from trainer import LangVaeTrainer
from vocab import SmilesTokenizer

import pandas as pd

import argparse
from omegaconf import OmegaConf
import yaml

import debugpy
debugpy.listen(('0.0.0.0', 5678))
debugpy.wait_for_client()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="vae_denovo")
    parser.add_argument("--wandb_entity", type=str, default="ty-kim")
    parser.add_argument("--wandb_key", type=str)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    return args

def yaml_load(file_path: str):
    f = open(file_path, 'r')
    data = yaml.safe_load(f)
    f.close()

    return data


def load_datalist(data_path):
    data = pd.read_csv(data_path)
    train_df = data[data['split'] == 'trn']
    train_list = train_df['smiles'].tolist()

    val_df = data[data['split'] == 'vld']
    val_list = val_df['smiles'].tolist()

    test_df = data[data['split'] == 'tst']
    test_list = test_df['smiles'].tolist()

    return train_list, val_list, test_list


if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    # Argument Setting
    # -----------------------------------------------------------------------------
    args = get_args()
    # cfg = yaml_load(args.config)
    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, False)

    # -----------------------------------------------------------------------------
    # loggder
    # -----------------------------------------------------------------------------
    try:
        import wandb
        wandb_avail = True
    except ImportError:
        wandb_avail = False

    if args.wandb and wandb_avail:
        wandb.login(key=args.wandb_key, force=True)
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)


    # -----------------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------------
    train_list, val_list, test_list = load_datalist(cfg.data_path)

    tokenizer = SmilesTokenizer(file_name=cfg.vocab_file_path)
    train_dataset = StringDataset(tokenizer, train_list)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=cfg.workers
    )

    for data in train_dataloader:
        print(data)
        break
    # -----------------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------------
    trainer = LangVaeTrainer(cfg)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    init_start_event.record()

    trainer.train()

    init_end_event.record()
    
    if args.wandb:
        wandb.finish()
    print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")