import torch
import numpy as np

import os

from vocab import SmilesTokenizer
from models import LSTMEncoder, LSTMDecoder, TransformerEncoder, TransformerDecoder, LangVAE

class VaeTrainer():
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def _build_model(self):
        if cfg.model_type == 'lstm':
            self.model = 
    
class LangVaeTrainer():
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = LangVAE(cfg)
        self.tokenizer = SmilesTokenizer(file_name=cfg.vocab_file_path)

    def train_step(self, data):
        nll_loss, mu, logvar = self.model.forward_by_tgt(data)
        loss, bce, kld = self.model.vae_loss(nll_loss, mu, logvar, beta=self.cfg.beta)

        return loss, bce, kld

    def eval_step(self, data):
        samples, _ = self.model.sample(100, greedy=False)
        stdmet = standard_metrics(sam_100, trn_set=[], subs_size=100)   # ignore novelty metric

        
    def train(
        self, 
        dataloader,
        epochs,
        save_freq,
        save_path,

    ):
        loss_collect = []
        for epoch in range(1, epochs+1):
            losses, bces, klds = [], [], []
            for idx, data in enumerate(dataloader):
                loss, bce, kld = self.train_step(data)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
                bces.append(bce.item())
                klds.append(kld.item())

        epo_loss = np.round([np.mean(losses), np.mean(bces), np.mean(klds)], decimals=4)
        loss_collect.append(epo_loss)

        if epoch % save_freq == 0:
            checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(), # 이렇게 해야 나중에 싱글 추론 가능
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict' : scheduler.state_dict()
                }
            torch.save(checkpoint, os.path.join(args.ckpt_dir, f'checkpoint_{epoch}.pth'))

        return loss_collect

            