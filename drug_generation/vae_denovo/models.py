import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from dataclasses import dataclass, asdict
from typing import List, Callable
from torch import nn, Tensor

class FFLayers(nn.Module):
    def __init__(
            self, 
            inp_size: int,
            ff_sizes: List[int],
            act_cls: Callable[..., nn.Module] = nn.ReLU,   # 클래스 자체를 받게!
            drop_p: float = 0.1,
            bnorm: bool = False,
    ):
        super().__init__()
        if not ff_sizes: # == if len(ff_sizes) == 0:
            raise ValueError("ff_sizes must not be empty")
        layers: List[nn.Module] = []
        in_feat = inp_size
    
        # hidden layers (모두 동일한 패턴)
        for out_feat in ff_sizes[:-1]:
            layers.append(nn.Linear(in_feat, out_feat))
            if bnorm:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(act_cls())
            if drop_p > 0.0:
                layers.append(nn.Dropout(drop_p))
            in_feat = out_feat

        layers.append(nn.Linear(in_feat, ff_sizes[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiLSTM(nn.Module):
    def __init__(
            self, 
            emb_size: int,
            voc_size: int,
            hidden_layer_units: List[int],
    ):
        super().__init__()
        if not hidden_layer_units:
            raise ValueError("hidden_layer_units must not be empty")
        
        self.embedding = nn.Embedding(voc_size, emb_size)

        self.lstm_list = nn.ModuleList()
        in_feat = emb_size
        for i in range(len(hidden_layer_units) - 1):
            self.lstm_list.append(nn.LSTMCell(in_feat, hidden_layer_units[i]))
            in_feat = hidden_layer_units[i]

        self.out_layer = nn.Linear(hidden_layer_units[-1], voc_size)
    
    def forward(self, x, hs, cs):
        emb_x = self.embedding(x) # emb_x.shape = (batch_size, feature_dim)
        hs[0], cs[0] = self.lstm_list[0](emb_x, (hs[0], cs[0]))
        for i in range(1, len(hs)):
            hs[i], cs[i] = self.lstm_list[i](hs[i-1], (hs[i], cs[i]))
        fc_out = self.linear(hs[len(hs)-1])
        return fc_out, hs, cs

# class EmbeddingLSTM():
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.lstm = MultiLSTM(
#             cfg.emb_size,
#             cfg.hidden_layer_units, 
#             cfg.voc_size
#         )
#         self.voc = cfg.voc
#         self.bosi = self.voc.get_bosi()
#         self.eosi = self.voc.get_eosi()
        

#     def unroll_target(self, target, init_hiddens=None, init_cells=None):
        

class LSTMEncoder():
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.lstm_embedder = EmbeddingLSTM(cfg)
        self.z_means = nn.Linear(cfg.embedding_size, cfg.latent_dim)
        self.z_var = nn.Linear(cfg.embedding_size, cfg.latent_dim)

        if cfg.embedding_size != self.lstm_embedder.ff_sizes[-1]:
            raise ValueError(f"Embedding size ({cfg.embedding_size}) does not match the final size of the LSTM ({self.lstm_embedder.ff_sizes[-1]})")

    def forward_by_tgt(self, tgt_seqs, conds={}):
        embedding = self.get_embed(tgt_seqs, conds)
        mu, logvar = self.z_means(embedding), self.z_var(embedding)
        repar = self.reparameterize(mu, logvar)
        return repar, mu, logvar, embedding
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_mem(self, tgt_seqs:torch.Tensor, conds={}):
        if conds == {}:
            conds['hs'], conds['cs'] = None, None
        mem = self.lstm_embedder.embed(
            tgt_seqs,
            init_hiddens=conds['hs'],
            init_cells=conds['cs']
        )
        return mem

class LSTMDecoder():
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_units = self.cfg.hidden_layer_units

        self.embedding = nn.Embedding(cfg.voc_size, cfg.embedding_size)

        self.lstm_list = nn.ModuleList()
        self.lstm_list.append(
            nn.LSTMCell(cfg.embedding_size, cfg.hidden_layer_units[0])
        )
        for i in range(1, len(self.hidden_units)):
            self.lstm_list.append(
                nn.LSTMCell(cfg.hidden_layer_units[i-1], cfg.hidden_layer_units[i])
            )

        self.latent_in = nn.Linear(cfg.latent_dim, cfg.embedding_size)
        self.linear = nn.Linear(self.hidden_units[-1], cfg.vocab_size)
        

    def get_params_groups(self):
        params_groups = [
            {"params": self.lstm_list.parameters()},
            {"params": self.linear.parameters()}
        ]
        return params_groups
    
    def decode2string(self, z, greedy=False, max_len=999, conds={}):
        seqs, _ = self.decode_z(z, greedy, max_len, conds)
        seqs_np = seqs.cpu().numpy()
        seqs_trcd = self.voc.truncate_eos(seqs_np)  # seqs truncated at EOS
        strings = ["".join(self.voc.decode(tok)) for tok in seqs_trcd]
        return strings, seqs


    def forward(self, x, z, hs, cs):
        # (batch_size, emb_size)
        emb_x = self.embedding(x)
        z = self.latent_in(z)

        # (batch_size, emb_sz * 2)
        lstm_input = torch.cat((emb_x, z), dim=1)

        hs[0], cs[0] = self.lstm_list[0](lstm_input, (hs[0], cs[0]))
        for i in range(1, len(hs)):
            hs[i], cs[i] = self.lstm_list[i](hs[i-1], (hs[i], cs[i]))
        fc_out = self.linear(hs[len(hs)-1])
        return fc_out, hs, cs
    
    def step_likelihood(self, xi, zs, hs, cs):
        logits, hidden_states, cell_states = self.forward(xi, zs, hs, cs)
        log_prob = F.log_softmax(logits, dim=1)
        prob = F.softmax(logits, dim=1)
        return prob, log_prob, logits, hidden_states, cell_states


    def forward_by_tgt_z(self, tgt_seqs, z, conds={}):
        """
        Args:
        conds: conds['hs']=hidden_states, conds['cs']=cell_states can be provided for conditions.
            If conds={}, then we just use None for default behavior(no condition).

        Returns:
        _prob_map: (bsz, tlen, voc_sz)
        likelihoods: (bsz)
        NLLLoss: (bsz)
        """
        bsz, tken = tgt_seqs.shape

        hidden_states, cell_states = [], []
        if conds == {}:
            for i in range(len(self.hidden_units)):
                hidden_states.append()
        else:
            for i in range(len(self.hl_units)):
                hidden_states.append(conds['hs'][i])
                cell_states.append(conds['cs'][i])

        start_token = torch.full((bsz,1), self.bosi).long().to(self.device)  # BOS column
        x = torch.cat((start_token, tgt_seqs[:, :-1]), dim=1)  # the last position of tgt_seqs won't be used for input.
        
        NLLLoss = tgt_seqs.new_zeros(bsz).float() 
        likelihoods = tgt_seqs.new_zeros(bsz, tlen).float()
        prob_map = tgt_seqs.new_zeros((bsz, self.voc.vocab_size, tlen)).float()

        for step in range(tlen):
            # Note that we are sliding a vertical scanner (height=batch_size) moving on timeline.
            x_step = x[:, step]  ## (batch_size)

            # let's find x_t[i] where it is <PAD>. Only <PAD>s will be True.
            padding_where = (_tgt_seqs[:, step] == self.padi)
            non_paddings = ~padding_where

            prob, log_prob, hidden_states, cell_states = self.step_likelihood(x_step, zvecs, hidden_states, cell_states)
            prob_map[:, :, step] = prob

            # the output of the lstm should be compared to the ones at x_step+1 (=target_step)
            one_hot_labels = nn.functional.one_hot(_tgt_seqs[:, step], num_classes=self.voc.vocab_size)

            # one_hot_labels.shape = (batch_size, vocab_size)
            # Make all the <PAD> tokens as zero vectors.
            one_hot_labels = one_hot_labels * non_paddings.reshape(-1,1)

            likelihoods[:, step] = torch.sum(one_hot_labels * prob, 1)
            loss = one_hot_labels * log_prob
            loss_on_batch = -torch.sum(loss, 1) # this is the negative log loss
            NLLLoss += loss_on_batch


        _prob_map = prob_map.transpose(1, 2)  # _prob_map: (bsz, tlen, voc_sz)
        return _prob_map, likelihoods, NLLLoss

    def decode_z(self, zvecs:torch.Tensor, greedy=False, max_len=999, conds={}):
        bsz, _ = zvecs.shape  # batch size and target length
        _zvecs = zvecs.to(self.device)

        hidden_states, cell_states = [], []
        if conds == {}:
            for i in range(len(self.hl_units)):
                hidden_states.append(_zvecs.new_zeros(bsz, self.hl_units[i]).float())
                cell_states.append(_zvecs.new_zeros(bsz, self.hl_units[i]).float())
        else:
            for i in range(len(self.hl_units)):
                hidden_states.append(conds['hs'][i])
                cell_states.append(conds['cs'][i])


        x_step = torch.full((bsz,), self.bosi).long().to(self.device)  # BOS tokens at the first step
        
        sequences, prob_list = [], []
        prob_map = _zvecs.new_zeros((bsz, self.voc.vocab_size, max_len)).float()
        finished = torch.zeros(bsz).byte() # memorize if the example is finished or not.

        for step in range(max_len):
            prob, _, hidden_states, cell_states = self.step_likelihood(x_step, _zvecs, hidden_states, cell_states)
            ## prob.shape = (bsz, vocab_size)
            prob_list.append(prob.view(bsz, 1, self.voc.vocab_size))

            if greedy == True:
                next_word = torch.argmax(prob, dim=1)
            else:
                next_word = torch.multinomial(prob, num_samples=1).view(-1)
            sequences.append(next_word.view(-1, 1))

            x_step = next_word.clone()  # next step input

            # is EOS sampled at a certain example?
            EOS_sampled = (next_word == self.eosi)
            finished = torch.ge(finished + EOS_sampled.cpu(), 1)
            # if all the examples have produced EOS once, we will break the loop
            if torch.prod(finished) == 1: break
        
        # Each element in sequences is in shape (bsz, 1)
        # concat on dim=1 to get (bsz, seq_len)
        bat_seqs = torch.cat(sequences, dim=1)
        # Each element in prob_list is in shape (bsz, 1, voc_sz)
        # concat on dim=1 to get (bsz, seq_len, voc_sz)
        prob_map = torch.cat(prob_list, dim=1)
        return bat_seqs, prob_map
        


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        two_i = torch.arange(0, d_model, 2)
        div = torch.exp(two_i * -(math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1,max_len,d_model)

    def forward(self, x: Tensor):          # x: (B,T,d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerEncoder(nn.Module):        # LangEncoder 대체
    def __init__(self, cfg):
        super().__init__()
        self.emb = nn.Embedding(cfg.voc_size, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model)
        enc_layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, dim_feedforward=4*cfg.d_model)
        self.encoder = nn.TransformerEncoder(enc_layer, cfg.num_layers)
        self.mem_pool = cfg.mem_pool          # 'cls' or 'mean'
        self.mu = nn.Linear(cfg.d_model, cfg.latent_dim)
        self.logvar = nn.Linear(cfg.d_model, cfg.latent_dim)

    def forward(self, token_ids):
        """
        token_ids: (B,T)
        """
        x = self.emb(token_ids) # (B,T,d_model)
        x = self.pos(x).transpose(0,1)     # (T,B,d_model) – PyTorch transformer expects seq‑first
        h = self.encoder(x)                # (T,B,d_model)

        if self.mem_pool == 'cls':
            mem = h[0]                     # 첫 토큰이 [CLS]
        else:
            mem = h.mean(dim=0)            # 평균 풀
        mu, logvar = self.mu(mem), self.logvar(mem)
        return mu, logvar                  # KL용

    
class TransformerDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.emb = nn.Embedding(vocab.vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        dec_layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=4*d_model)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.z2mem  = nn.Linear(d_latent, d_model)    # z → memory vector
        self.proj = nn.Linear(d_model, vocab.vocab_size)
        self.bosi, self.eosi, self.padi = vocab.get_bosi(), vocab.get_eosi(), vocab.get_padi()
    
    def _prep_masks(self, tgt_len, src_len, device):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)
        src_key_padding_mask = None                    # z 는 길이 1 ⇒ 마스크 필요X
        return tgt_mask, src_key_padding_mask

    # teacher forcing
    def forward_by_tgt_z(self, tgt, z):
        """
        tgt: (B,T)  – <eos>, <pad> 포함, <bos> 미포함
        z  : (B,d_latent)
        """
        B,T = tgt.shape
        device = tgt.device
        mem = self.z2mem(z).unsqueeze(0)               # (1,B,d_model)
        tgt_inp = torch.cat([torch.full((B,1), self.bosi, device=device),   # prepend <bos>
                             tgt[:,:-1]], dim=1)
        x = self.pos(self.emb(tgt_inp)).transpose(0,1) # (T,B,d_model)
        tgt_mask, _ = self._prep_masks(T, 1, device)
        out = self.decoder(tgt=x, memory=mem,
                           tgt_mask=tgt_mask)
        logits = self.proj(out.transpose(0,1))         # (B,T,V)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # NLL (teacher forcing, ignore PAD)
        nll = nn.functional.nll_loss(
              log_probs.reshape(-1, self.vocab.vocab_size),
              tgt.reshape(-1), ignore_index=self.padi, reduction='none')
        nll = nll.view(B, T).sum(dim=1)                # (B,)

        return probs, None, nll                        # compat: prob_map·likelihoods 생략 OK

    # sampling/greedy decoding
    @torch.no_grad()
    def decode_z(self, z, greedy=False, max_len=200):
        B = z.size(0); device = z.device
        mem = self.z2mem(z).unsqueeze(0)               # (1,B,d_model)
        ys  = torch.full((B,1), self.bosi, device=device, dtype=torch.long)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_len):
            x = self.pos(self.emb(ys)).transpose(0,1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(device)
            out = self.decoder(x, mem, tgt_mask=tgt_mask)
            logits = self.proj(out[-1])                # last step (B,V)
            next_tok = (logits.argmax(-1) if greedy
                        else torch.multinomial(nn.functional.softmax(logits, -1), 1).squeeze(-1))
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            finished |= (next_tok == self.eosi)
            if finished.all(): break
        return ys[:,1:], None                          # strip <bos>
        
class LangVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.build_models()

    def build_models(self):
        if self.cfg.model_type == 'lstm':
            self.encoder = LSTMEncoder(self.cfg)
            self.decoder = LSTMDecoder(self.cfg)
        elif self.cfg.model_type == 'transformer':
            self.encoder = TransformerEncoder(self.cfg)
            self.decoder = TransformerDecoder(self.cfg)
        else:
            raise ValueError(f"Unknown model type: {self.cfg.model_type}")

    def forward_by_tgt(self, data):
        repar, mu, logvar, mem = self.encoder.forward_by_tgt(data)
        pmap, likelihoods, nll_loss = self.decoder.forward_by_tgt_z(data, repar)

        return nll_loss, mu, logvar
    
    def vae_loss(self, nll_loss, mu, logvar, beta=1.0):
        """
        Binary Cross Entropy Loss + KL Divergence 
        nll_loss: negative log likelihood loss
        mu: mean of the latent variable
        logvar: log variance of the latent variable
        beta: weight for the KL divergence term
        """
        bce = torch.mean(nll_loss)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        if torch.any(torch.isnan(kld)):
            print(kld)
        beta_kld = beta * kld
        return bce + beta_kld, bce, kld
        

    def reconstruct(self, dataloader):
        recon_strs, inp_bats, recon_bats  = [], [], []
        for idx, data in enumerate(dataloader):
            inp_bats.append(data)
            repar, mu, logvar, mem = self.encoder.forward_by_tgt(data)
            # reconstruction is done by zvec = mu, i.e. zero variance
            zvecs = mu
            strs, seqs = self.decoder.decode2string(zvecs, greedy=True, max_len=self.conf.max_len)
            recon_bats.append(seqs.cpu())  # output batches
            recon_strs.append(strs)  # reconstructed strings, item is a batch

        _recon_strs = []  # flatten the list of list
        for slist in recon_strs:
            _recon_strs += slist
        return _recon_strs, inp_bats, recon_bats
    

    def sample(self, n_samples):
        sampled_bats, zvec_bats = [], []  # batches are collected
        cnt = 0
        while cnt < n_samples:
            z = torch.randn((self.cfg.batch_size, self.cfg.latent_dim))
            zvec_bats.append(z)
            gen_strs, _ = self.decoder.decode2string(z, greedy, self.conf.max_len)
            sampled_bats.append(gen_strs)
            cnt += self.cfg.batch_size

        samples = []  # flatten the list of list
        for slist in sampled_bats:
            samples += slist
        sam_zvecs = torch.vstack(zvec_bats)
        return samples[:n_samples], sam_zvecs[:n_samples]
    

