import os
import argparse
import yaml
import wandb


from torch_geometric.loader import DataLoader

from lightning.pytorch.loggers import WandbLogger
from config import Config
from utils import convert_data



class TGVAETrainer:
    def __init__(self, config, vocab_smi):
        self.cfg = config
        self.vocab_smi = vocab_smi
        
        
    def build_model_optimizer(self):

        # Optimizer
        params_to_optimize = []
        params = self._get_trainable_params(do_save_log_txt=False, do_print=False)
        params_to_optimize.append({"params": params, "lr": self.cfg.lr})

        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.cfg.lr,
            weight_decay=0.01
        )


    def loss_fn(self, out, tgt, beta, config, weight=None):
        pred, mu, sigma = out
        recon_loss = F.nll_loss(pred.reshape(-1, len(self.vocab_smi)), tgt.reshape(-1), ignore_index=self.vocab_smi['[PAD]'], weight=weight)
        
        if loss_kl=='mean':
            kl_loss = -0.5 * torch.mean(1 + sigma - mu.pow(2) - sigma.exp())
        elif loss_kl=='sum':
            kl_loss = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp()) / mu.size(0)

        return recon_loss + beta * kl_loss


    def train_step(self, batch):
        inp_graph, inp_smi, inp_smi_mask, tgt_smi = convert_data(batch, self.vocab_smi, device=device)
        output = self.model(inp_graph, inp_smi, inp_smi_mask)
        loss = self.loss_fn(output, tgt_smi, annealer[e-1], config)

        return loss


    def train(self):
        for e in range(self.cfg.trained_epoch + 1, self.cfg.epoch + 1):
            self.model.train()
            self.trainloader = tqdm(self.trainloader) if self.is_master_rank else self.trainloader
            for batch in self.trainloader:
                with te.fp8_autocast(enabled=self.is_transformer_engine, fp8_recipe=fp8_recipe, fp8_group=dist.group.WORLD):
                    with torch.autocast(device_type='cuda', dtype=self.weight_dtype, enabled=True):
                        loss_dict = self.train_step(batch)
                        loss.backward()

            if config.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
            optim.step()
            optim.zero_grad()
            
            if e % config.save_every == 0 and e >= config.start_save:
                checkpoint(model, optim, e, config)
            if e % config.generate_every == 0 and e >= config.start_generate:
                generate_molecule(model, config, NUM_GEN, op.join(config.path_generate_folder, f'epoch_{e}.txt'), batch=BATCH_GEN)

if __name__ == '__main__':
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
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
    model = TGVAE(config)
    model.train()