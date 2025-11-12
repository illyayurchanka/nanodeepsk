"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Multi-Query Attention (MQA) support for more efficient inference
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW


@dataclass
class GPTConfig:
    sequence_len: int = 10
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 16  # number of query heads
    head_dim: int = 448
    n_embd: int = 7168

    q_compression = 576
    kv_compression = 576 * 2
    rotate_dim = 576


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last time into two halves
    y1 = x1 * cos + x2 * sin  # rotate_dim pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)  # re-assemble
    out = out.to(x.dtype)  # ensure input/output dtypes match
    return out


class LatentSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        self.layer_idx = layer_idx

        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.head_dim

        assert self.n_embd % self.n_head == 0, (
            "embedding dimension must be divisible by number of heads"
        )

        self.q_compression = config.q_compression
        self.kv_compression = config.kv_compression
        assert self.q_compression < self.n_embd and self.kv_compression < self.n_embd, (
            "compression ratio must be less than embedding dimension"
        )

        self.rotate_dim = config.rotate_dim

        # compression projection
        # query projection
        self.Wq_d = nn.Linear(self.n_embd, self.q_compression)
        # key-value projection
        self.Wkv_d = nn.Linear(self.n_embd, self.kv_compression, bias=False)

        # up projections
        # query up projection
        self.Wq_u = nn.Linear(
            self.q_compression, self.n_head * self.head_dim, bias=False
        )
        # key up projection
        self.Wk_u = nn.Linear(
            self.kv_compression, self.head_dim * self.n_head, bias=False
        )
        # value up projection
        self.Wv_u = nn.Linear(
            self.kv_compression, self.head_dim * self.n_head, bias=False
        )

        # rotation projection
        # query rotation projection
        self.W_qr = nn.Linear(
            self.q_compression, self.n_head * self.rotate_dim, bias=False
        )
        # key rotation projection
        self.W_kr = nn.Linear(self.n_embd, self.rotate_dim, bias=False)

        # output projection
        self.W_out = nn.Linear(self.n_head * self.head_dim, self.n_embd, bias=False)

        # precomputed (for eval mode)
        # precomputed matrix multiplication of weight_q and weigth_k for multiple heads
        self.W_qk = None  # nn.Linear(
        # self.W_qk = nn.Linear(
        #     self.q_compression, self.n_head * self.kv_compression, bias=False
        # )
        # )
        # output transformation (precomputed multiplication of weight_v and weight_out)
        # self.W_out_prime = nn.Linear(
        #     self.kv_compression * self.n_head, self.n_embd, bias=False
        # )

        self.scaler = float(
            1.0 / math.sqrt(self.kv_compression + self.rotate_dim)
        )  # Store as float in initialization
        self.device = (
            torch.device("cuda", 0)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    def forward(self, x, cos_sin=None, kv_cache=None):
        # batch_size, seq_len, embedding_dim
        B, T, C = x.size()

        assert C == self.n_embd, (
            f"Input embedding dimension {C} does not match expected dimension {self.n_embd}"
        )

        # Query compressed
        cq_t = self.Wq_d(x)  # [B, T, q_compression]
        # KeyValue compressed
        ckv_t = self.Wkv_d(x)  # [B, T, kv_compression]
        train = False
        # Query uncompressed
        if train:
            Q_t = self.Wq_u(cq_t).view(
                B, self.n_head, T, self.head_dim
            )  # [B, n_head, T, head_dim]
            V = self.Wv_u(ckv_t).view(B, T, self.head_dim)  # [B, T, head_dim]
            K_t = self.Wk_u(ckv_t).view(B, T, self.head_dim)  # [B, T, head_dim] #
        else:
            if self.W_qk is None:
                self._absorb()
            Q_t = self.W_qk(cq_t).view(
                B, self.n_head, T, self.kv_compression
            )  # [B, n_head, T, kv_compression]
            V = ckv_t
            K_t = ckv_t

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        if cos_sin is not None:
            cos, sin = cos_sin
            K_r, Q_r = (
                apply_rotary_emb(self.W_kr(x), cos, sin),  # [B, T, rotate_dim]
                apply_rotary_emb(self.W_qr(cq_t), cos, sin).view(
                    B, self.n_head, T, self.rotate_dim
                ),  # [B, n_head, T, rotate_dim]
            )  # QK rotary embedding
        else:
            K_r, Q_r = (
                self.W_kr(x),
                self.W_qr(cq_t).view(B, self.n_head, T, self.rotate_dim),
            )  # no embedding (just checking shapes)
        # q, k = norm(q), norm(k)  # QK norm

        # Cache
        if kv_cache is not None:
            K_t, K_r = kv_cache.insert_kv(self.layer_idx, K_t, K_r)

        Q = torch.cat([Q_t, Q_r], dim=-1)  # [B, n_head, T, head_dim + rotate_dim]

        K = torch.cat([K_t, K_r], dim=-1)  # [B, T, head_dim + rotate_dim]

        Tq = Q.size(2)
        Tk = K.size(2)

        K, V = (
            K.unsqueeze(1),  # [B, 1, T, head_dim + rotate_dim]
            V.unsqueeze(1),  # [B, 1, T, head_dim]
        )
        if kv_cache is None:
            attention = F.scaled_dot_product_attention(
                Q, K, V, is_causal=True, scale=self.scaler
            )  # [B, num_heads, T, head_dim]
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            attention = F.scaled_dot_product_attention(
                Q, K, V, is_causal=False, scale=self.scaler
            )
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros(
                (Tq, Tk), dtype=torch.bool, device=q.device
            )  # True = keep, False = mask
            prefix_len = Tk - Tq
            if prefix_len > 0:  # can't be negative but could be zero
                attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(
                torch.ones((Tq, Tq), dtype=torch.bool, device=q.device)
            )
            attention = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=attn_mask, scale=self.scaler
            )
        # Re-assemble the heads side by side and project back to residual stream
        attention = (
            attention.transpose(1, 2).contiguous().view(B, T, -1)
        )  # [B, num_heads, T, kv_compression] ->  [B, T, num_heads, head_dim] -> [B, T, num_heads * head_dim]
        print(f"attention.shape: {attention.shape}")
        if train:
            output = self.W_out(attention)
        else:
            output = self.W_out_prime(attention)

        return output

    def _absorb(self):
        """
        Create absorbed weights for inference:
        W_q' = W_q @ W_uk^T
        W_o' = W_uv @ W_o
        Returns two Linear layers that replace (W_q, W_uk) and (W_uv, W_o).
        """
        # Extract raw weight matrices (note .weight is (out_features, in_features))
        Wq_u = self.Wq_u.weight.data  # (d_all, d_model)
        Wk_u = self.Wk_u.weight.data  # (d_all, d_c)
        Wv_u = self.Wv_u.weight.data  # (d_all, d_c)
        W_out = self.W_out.weight.data  # (d_model, d_all)

        # Compute fused weights with shapes matching Linear(in=d_model, out=d_c) and (in=d_c, out=d_model)
        print(Wq_u.shape)
        print(Wk_u.shape)
        W_qk = (Wq_u.t() @ Wk_u).t().contiguous()  # (q_compressd, kv_compressed)

        W_out_prime = (Wv_u.t() @ W_out).t().contiguous()  # (kv_compressed, n_embed)
        print(W_out_prime.shape2)
        # Build small Linear layers holding fused weights
        self.W_qk = nn.Linear(
            self.q_compression, self.kv_compression, device=self.device, bias=False
        )
        self.W_out_prime = nn.Linear(
            self.kv_compression * self.n_head, self.n_embd, bias=False
        )
        self.W_qk.weight.data.copy_(W_qk)
        self.W_out_prime.weight.data.copy_(W_out_prime)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx=1):
        super().__init__()
        self.attn = LatentSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin=None, kv_cache=None):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList(
                    [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
                ),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = (
            config.sequence_len * 10
        )  # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer(
            "cos", cos, persistent=False
        )  # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()  # keep them in bfloat16
        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        )  # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311"""
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = (
            self.config.n_layer,
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.sequence_len,
        )
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(
        self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0
    ):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(
            embedding_params
        ) + len(lm_head_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(
                f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}"
            )
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim))
        assert T <= self.cos.size(1), (
            f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        )
        assert idx.device == self.cos.device, (
            f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        )
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = (
            self.cos[:, T0 : T0 + T],
            self.sin[:, T0 : T0 + T],
        )  # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15
        if targets is not None:
            # training mode: compute and return the loss
            # TODO: experiment with Liger Kernels / chunked cross-entropy etc.
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)  # logits softcap
            logits = logits.float()  # use tf32/fp32 for logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss
        else:
            # inference mode: compute and return the logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)  # logits softcap
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
