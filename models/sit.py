# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################            
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
            )
        if "fused_attn" in block_kwargs.keys():
            self.attn.fused_attn = block_kwargs["fused_attn"]
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class VWNBlock(nn.Module):
    """
    Virtual Width Network wrapper around a backbone transformer block.

    Implements Generalized Hyper-Connections (GHC) as in Eq. (6) and Algorithm 1
    of the VWN paper, using the given SiTBlock as the backbone layer T_l.

    Shapes per token:
        - Backbone width D (hidden_size)
        - Fraction parameter m, expanded width n (r = n/m)
        - Per-block width db = D / m
        - Over-width dim D' = n * db
        - Over-width hidden state H' ∈ R^{n × db}
    """
    def __init__(self, backbone_block: nn.Module, hidden_size: int,
                 m: int = 2, n: int = 3, use_dynamic: bool = True):
        super().__init__()
        assert hidden_size % m == 0, "hidden_size must be divisible by m"
        assert n > m, "Require n > m for non-trivial virtual width"

        self.backbone = backbone_block  # SiTBlock (T_l)
        self.D = hidden_size
        self.m = m
        self.n = n
        self.db = hidden_size // m
        self.D_prime = self.n * self.db
        self.use_dynamic = use_dynamic

        # Static routing matrices B ∈ R^{m×n}, A ∈ R^{n×(m+n)}
        self.B = nn.Parameter(torch.zeros(self.m, self.n))
        self.A = nn.Parameter(torch.zeros(self.n, self.m + self.n))

        if self.use_dynamic:
            # Scaling matrices Sβ ∈ R^{m×n}, Sα ∈ R^{n×(m+n)}
            self.S_beta = nn.Parameter(torch.ones(self.m, self.n))
            self.S_alpha = nn.Parameter(torch.ones(self.n, self.m + self.n))

            # Projection weights Wβ ∈ R^{db×m}, Wα ∈ R^{db×(m+n)}
            self.W_beta = nn.Parameter(torch.zeros(self.db, self.m))
            self.W_alpha = nn.Parameter(torch.zeros(self.db, self.m + self.n))

            # Slot-wise normalization over the per-block width db
            self.slot_norm = nn.LayerNorm(self.db)
        else:
            self.register_parameter("S_beta", None)
            self.register_parameter("S_alpha", None)
            self.register_parameter("W_beta", None)
            self.register_parameter("W_alpha", None)
            self.slot_norm = nn.Identity()

        self._init_static_matrices()

    def _init_static_matrices(self):
        """Initialize static B and A as in Eqs. (11) and (12)."""
        with torch.no_grad():
            # Eq. (11): cyclic pattern for B ∈ R^{m×n}
            for i in range(self.m):
                for j in range(self.n):
                    if i == (j % self.m):
                        self.B[i, j] = 1.0

            # Eq. (12): block structure for A ∈ R^{n×(m+n)}
            A = torch.zeros(self.n, self.m + self.n)
            m, n = self.m, self.n
            r = n - m
            if n == m:
                # [I_m  I_m]
                A[:m, :m] = torch.eye(m)
                A[:m, m:2*m] = torch.eye(m)
            else:
                # Top block: [I_m  I_m  0_{m×r}]
                A[:m, :m] = torch.eye(m)
                A[:m, m:2*m] = torch.eye(m)
                # Bottom block: [0_{r×m} 0_{r×m} I_r]
                if r > 0:
                    A[m:, 2*m:2*m + r] = torch.eye(r)
            self.A.copy_(A)

    def _compute_dynamic_AB(self, H_flat):
        """
        Compute dynamic A(H') and B(H') for a batch of tokens.

        Args:
            H_flat: (BT, n, db) where BT = batch_size * seq_len

        Returns:
            A_t: (BT, n, m+n)
            B_t: (BT, m, n)
        """
        if not self.use_dynamic:
            BT = H_flat.shape[0]
            A_t = self.A.unsqueeze(0).expand(BT, -1, -1)
            B_t = self.B.unsqueeze(0).expand(BT, -1, -1)
            return A_t, B_t

        # Eq. (8): normalize H' (slot-wise over db)
        H_norm = self.slot_norm(H_flat)  # (BT, n, db)
        tau = math.sqrt(self.db)

        # Eq. (9): B(H')
        # H_norm @ Wβ: (BT, n, m)
        B_delta = torch.tanh((H_norm @ self.W_beta) / tau)  # (BT, n, m)
        B_delta = B_delta.transpose(1, 2)                   # (BT, m, n)
        B_t = self.B + self.S_beta * B_delta                # broadcast

        # Eq. (10): A(H')
        # H_norm @ Wα: (BT, n, m+n)
        A_delta = torch.tanh((H_norm @ self.W_alpha) / tau)  # (BT, n, m+n)
        A_t = self.A + self.S_alpha * A_delta                # broadcast

        return A_t, B_t

    def forward(self, H_prev, c):
        """
        Args:
            H_prev: (N, T, n, db)  Over-width hidden states at layer l-1
            c:      (N, D)        Conditioning (t_embed + y_embed)

        Returns:
            H_new:  (N, T, n, db) Updated over-width hidden states H'_l
            z:      (N, T, D)     Backbone output z_l (for projectors, etc.)
        """
        N, T, n, db = H_prev.shape
        assert n == self.n and db == self.db
        BT = N * T

        # Flatten batch and time for token-wise computation
        H_flat = H_prev.view(BT, n, db)           # (BT, n, db)

        # Dynamic (or static) routing matrices
        A_t, B_t = self._compute_dynamic_AB(H_flat)
        # A_t: (BT, n, m+n), B_t: (BT, m, n)

        # Width connection: A_l^T H'_{l-1} ∈ R^{(m+n)×db}
        AH = torch.bmm(A_t.transpose(1, 2), H_flat)   # (BT, m+n, db)

        # First m rows → compressed backbone input X_l ∈ R^{m×db}
        X = AH[:, :self.m, :]                         # (BT, m, db)
        # Last n rows → A_hat^T H'_{l-1} (carry/forget operator)
        carry = AH[:, self.m:, :]                     # (BT, n, db)

        # Reshape X_l to D and feed into backbone block T_l
        x_in = X.reshape(N, T, self.D)                # (N, T, D)
        z = self.backbone(x_in, c)                    # (N, T, D)

        # Reshape backbone output into m segments of size db
        Z = z.view(BT, self.m, db)                    # (BT, m, db)

        # Depth connection: B_l^T Z_l + A_hat_l^T H'_{l-1}
        H_new = torch.bmm(B_t.transpose(1, 2), Z) + carry  # (BT, n, db)

        H_new = H_new.view(N, T, n, db)
        return H_new, z


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return x


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        path_type='edm',
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        decoder_hidden_size=768,
        encoder_depth=8,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_cfg=False,
        z_dims=[768],
        projector_dim=2048,
        # --- NEW: VWN options ---
        vwn_enabled: bool = False,
        vwn_m: int = 2,
        vwn_n: int = 3,
        vwn_dynamic: bool = True,
        **block_kwargs  # fused_attn, qk_norm, ...
    ):
        super().__init__()
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_cfg = use_cfg
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.encoder_depth = encoder_depth

        # --- VWN flags ---
        self.vwn_enabled = vwn_enabled
        self.vwn_m = vwn_m
        self.vwn_n = vwn_n
        self.vwn_dynamic = vwn_dynamic

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
            )
        self.t_embedder = TimestepEmbedder(hidden_size) # timestep embedding type
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # Backbone blocks (always SiTBlock)
        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs)
            for _ in range(depth)
        ])

        # --- NEW: VWN wrappers around blocks ---
        if self.vwn_enabled:
            assert hidden_size % self.vwn_m == 0, "hidden_size must be divisible by vwn_m"
            assert self.vwn_n > self.vwn_m, "Require vwn_n > vwn_m for non-trivial virtual width"

            self.vwn_db = hidden_size // self.vwn_m
            self.vwn_D_prime = self.vwn_n * self.vwn_db  # over-width dimension D'

            # Over-width expansion (Eq. (3): Ewide = W_expand E_base)
            self.vwn_expand = nn.Linear(hidden_size, self.vwn_D_prime)

            # GHC layers wrapping each backbone block
            self.vwn_layers = nn.ModuleList([
                VWNBlock(block, hidden_size,
                         m=self.vwn_m, n=self.vwn_n, use_dynamic=self.vwn_dynamic)
                for block in self.blocks
            ])

            # Reduce operator (Eq. (4)): W_reduce ∈ R^{D×D'}
            self.vwn_reduce_norm = nn.LayerNorm(self.vwn_D_prime)
            self.vwn_reduce = nn.Linear(self.vwn_D_prime, hidden_size)
        else:
            self.vwn_expand = None
            self.vwn_layers = None
            self.vwn_reduce_norm = None
            self.vwn_reduce = None

        self.projectors = nn.ModuleList([
            build_mlp(hidden_size, projector_dim, z_dim) for z_dim in z_dims
        ])
        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
    
    def forward(self, x, t, y, return_logvar=False):
        """
        Forward pass of SiT.

        x: (N, C, H, W) tensor of spatial inputs (images or latent representations)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # Patch embedding + fixed 2D sin-cos position embedding
        x_tokens = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        N, T, D = x_tokens.shape

        # Timestep and class embeddings -> conditioning vector c
        t_embed = self.t_embedder(t)              # (N, D)
        y_embed = self.y_embedder(y, self.training)  # (N, D)
        c = t_embed + y_embed                     # (N, D)

        zs = None

        if not self.vwn_enabled:
            # -------- Standard SiT path (unchanged) --------
            x_hid = x_tokens  # (N, T, D)
            for i, block in enumerate(self.blocks):
                x_hid = block(x_hid, c)  # (N, T, D)
                if (i + 1) == self.encoder_depth:
                    zs = [projector(x_hid.reshape(-1, D)).reshape(N, T, -1)
                          for projector in self.projectors]

            x_out_tokens = self.final_layer(x_hid, c)   # (N, T, patch_size**2 * out_channels)

        else:
            # -------- VWN path: Over-width embedding + GHC --------

            # Over-width embedding: Ewide = W_expand Ebase  (Eq. (3))
            # x_tokens: (N, T, D) -> e: (N, T, D')
            e = self.vwn_expand(x_tokens)  # (N, T, D')
            # Reshape into slot matrix H'_0 per token: (n, D'/n) with D'/n = db = D/m
            H = e.view(N, T, self.vwn_n, self.vwn_db)  # (N, T, n, db)

            for i, vwn_block in enumerate(self.vwn_layers):
                H, x_hid = vwn_block(H, c)  # H: (N, T, n, db), x_hid: (N, T, D)
                if (i + 1) == self.encoder_depth:
                    zs = [projector(x_hid.reshape(-1, D)).reshape(N, T, -1)
                          for projector in self.projectors]

            # Reduce last over-width hidden states back to width D (Eq. (4))
            # H_L: (N, T, n, db) -> (N*T, D')
            H_flat = H.view(N * T, self.vwn_D_prime)
            H_norm = self.vwn_reduce_norm(H_flat)
            h_L = self.vwn_reduce(H_norm)         # (N*T, D)
            x_for_final = h_L.view(N, T, D)       # (N, T, D)

            x_out_tokens = self.final_layer(x_for_final, c)  # (N, T, patch_size**2 * out_channels)

        # Unpatchify
        x_out = self.unpatchify(x_out_tokens)  # (N, out_channels, H, W)
        return x_out, zs


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}

