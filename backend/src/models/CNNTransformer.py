import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# Audio / CQT
SR = 44_100
HOP = 384
FPS = SR / HOP
BINS_PER_OCTAVE = 36
N_OCTAVES = 7
N_BINS = BINS_PER_OCTAVE * N_OCTAVES

# MIDI label matrix
MIDI_LO = 21  # A0
MIDI_HI = 108 # C8
N_PITCHES = MIDI_HI - MIDI_LO + 1
CH_ACTIVE = 0
CH_ONSET = 1
CH_VEL = 2
N_LABEL_CHANNELS = 3

# Dataset build
SPLIT = "train"
LIMIT = 10  # Use an integer for smoke tests, or None for the full split.
CHUNK_FRAMES = 384
CHUNK_HOP_FRAMES = 384
ONSET_RADIUS = 1
KEEP_INCOMPLETE = False
MAX_GAP_FRAMES = 43
TARGET_MODE = "active_onset"
TARGET_CHANNELS = ("active", "onset")
N_TARGET_CHANNELS = len(TARGET_CHANNELS)

# Harmonic stacking
class HarmonicStacking(nn.Module):
    """
    Takes CQT input (B, 1, T, F) and stacks shifted frequency views.

    Each harmonic ratio h corresponds to a shift of:
        bins_per_octave * log2(h)

    Example ratios:
        [0.5, 1, 2, 3, 4, 5, 6, 7]
    """
    def __init__(
        self,
        bins_per_octave: int,
        harmonics=(0.5, 1, 2, 3, 4, 5, 6, 7),
        target_bins: int | None = None,
    ):
        super().__init__()
        self.bins_per_octave = bins_per_octave
        self.harmonics = harmonics
        self.target_bins = target_bins

        self.shifts = [
            int(round(bins_per_octave * math.log2(h)))
            for h in harmonics
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, F)
        assert x.ndim == 4
        B, C, T, Freq = x.shape
        assert C == 1

        stacked = []

        for shift in self.shifts:
            # For candidate pitch bin p, we want to read energy at p + shift.
            # Positive shift means harmonic is higher in frequency.
            if shift > 0:
                shifted = F.pad(x[..., shift:], (0, shift))
            elif shift < 0:
                shifted = F.pad(x[..., :shift], (-shift, 0))
            else:
                shifted = x

            stacked.append(shifted)

        # (B, H, T, F)
        x = torch.cat(stacked, dim=1)

        if self.target_bins is not None:
            x = x[..., :self.target_bins]

        return x



# ── CNN frontend ──────────────────────────────────────────────────────────────
 
class ConvBlock(nn.Module):
    """Conv2d → BN → GELU, no time-axis stride ever."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
 
 
class CNNFrontend(nn.Module):
    """
    Maps (B, 1, T, F=N_BINS) → (B, d_model, T, 88).
 
    Strategy
    --------
    Three conv blocks progressively expand channels.
    A final adaptive pool along the frequency axis collapses F → 88.
    Time axis is never touched.
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.hstack = HarmonicStacking(
            bins_per_octave=BINS_PER_OCTAVE,
            harmonics=(0.5, 1, 2, 3, 4, 5, 6, 7),
            target_bins=N_BINS,
        )

        self.convs = nn.Sequential(
            ConvBlock(8, 32),
            ConvBlock(32, 64),
            ConvBlock(64, d_model),
        )
        # Learned projection: F=252 → 88 along the frequency axis
        # AdaptiveAvgPool2d(output_size=(T, 88)) would pool time too,
        # so we pool frequency only with a 1D adaptive pool.
        self.freq_proj = nn.Sequential( # This could also be a Conv1D 
            nn.Linear(N_BINS, N_PITCHES),
            nn.GELU(),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hstack(x)       # (B, 8, T, F)
        x = self.convs(x)        # (B, C, T, F)
        x = self.freq_proj(x)    # (B, C, T, 88)
        return x
 
 
# ── positional embeddings ─────────────────────────────────────────────────────
 
class AxialPositionalEmbedding(nn.Module):
    """
    Learned additive embeddings along time and pitch axes, broadcast over
    the other axis so they can simply be summed together.
 
        time  embed: (1, T_max, 1,  C)
        pitch embed: (1, 1,     88, C)
    """
    def __init__(self, d_model: int, max_frames: int = 512):
        super().__init__()
        self.time_emb  = nn.Embedding(max_frames, d_model)
        self.pitch_emb = nn.Embedding(N_PITCHES,    d_model)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 88, C)
        B, T, P, C = x.shape
        t_idx = torch.arange(T, device=x.device)          # (T,)
        p_idx = torch.arange(P, device=x.device)          # (88,)
        t_emb = self.time_emb(t_idx).unsqueeze(1)         # (T, 1,  C)
        p_emb = self.pitch_emb(p_idx).unsqueeze(0)        # (1, 88, C)
        return x + t_emb + p_emb                          # broadcast over B
 
 
# ── axial attention blocks ────────────────────────────────────────────────────
 
class TimeAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = SDPASelfAttention(d_model, n_heads, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, P, C = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B * P, T, C)
        h = self.attn(h)
        h = h.reshape(B, P, T, C).permute(0, 2, 1, 3)
        return self.norm(x + h)


class PitchAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = SDPASelfAttention(d_model, n_heads, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, P, C = x.shape
        h = x.reshape(B * T, P, C)
        h = self.attn(h)
        h = h.reshape(B, T, P, C)
        return self.norm(x + h)
 
 
class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(0.1),
        )
        self.norm = nn.LayerNorm(d_model)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))
 
 
class AxialBlock(nn.Module):
    """One axial transformer block: TimeAttn → PitchAttn → FFN."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.time_attn  = TimeAttention(d_model, n_heads)
        self.pitch_attn = PitchAttention(d_model, n_heads)
        self.ffn        = FeedForward(d_model)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.time_attn(x)
        x = self.pitch_attn(x)
        x = self.ffn(x)
        return x

class SDPASelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Bseq, L, C)
        Bseq, L, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(Bseq, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, Bseq, H, L, D)

        q, k, v = qkv[0], qkv[1], qkv[2]

        dropout_p = self.dropout if self.training else 0.0

        h = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=False,
        )

        h = h.transpose(1, 2).contiguous().view(Bseq, L, C)
        return self.out(h)


 
# ── full model ────────────────────────────────────────────────────────────────
 
class PianoTranscriber(nn.Module):
    """
    Full AMT model.
 
    Parameters
    ----------
    d_model      : int   transformer / CNN channel width  (default 128)
    n_heads      : int   attention heads                  (default 4)
    n_layers     : int   number of axial blocks           (default 4)
    max_frames   : int   maximum sequence length          (default 512)
 
    Forward
    -------
    x : (B, 1, T, N_BINS)  log-magnitude CQT chunk
    → logits : (B, T, 88, 2)
        [:,:,:,0]  active logit  (apply sigmoid for probability)
        [:,:,:,1]  onset  logit
 
    Loss (computed externally)
    --------------------------
    Both heads use BCE with logits.
    Onset loss weighted higher (onsets are sparse — use pos_weight or focal).
    """
 
    def __init__(
        self,
        d_model:    int = 128,
        n_heads:    int = 4,
        n_layers:   int = 4,
        max_frames: int = 512,
    ):
        super().__init__()
 
        self.cnn     = CNNFrontend(d_model)
        self.proj    = nn.Linear(d_model, d_model)   # channel mixer after CNN
        self.pos_emb = AxialPositionalEmbedding(d_model, max_frames)
 
        self.blocks  = nn.ModuleList([
            AxialBlock(d_model, n_heads) for _ in range(n_layers)
        ])
 
        self.head = nn.Linear(d_model, N_TARGET_CHANNELS)        # → 2 logits per (t, pitch)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── CNN ───────────────────────────────────────────────────────────────
        x = self.cnn(x)                              # (B, C, T, 88)
 
        # ── rearrange to (B, T, 88, C) for transformer ────────────────────────
        x = x.permute(0, 2, 3, 1)                   # (B, T, 88, C)
        x = self.proj(x)                             # learned channel mix
 
        # ── positional embeddings ─────────────────────────────────────────────
        x = self.pos_emb(x)
 
        # ── axial transformer blocks ──────────────────────────────────────────
        for block in self.blocks:
            x = block(x)
 
        # ── output head ───────────────────────────────────────────────────────
        return self.head(x)                          # (B, T, 88, 2)
 
 
# ── loss ──────────────────────────────────────────────────────────────────────
 
def amt_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    active_pos_weight: float = 35.0,
    onset_pos_weight: float = 200.0,
    onset_loss_weight: float = 2.0,
) -> tuple[torch.Tensor, dict]:

    active_logits = logits[..., 0]
    onset_logits  = logits[..., 1]

    active_target = targets[..., 0]
    onset_target  = targets[..., 1]

    active_pw = torch.tensor(active_pos_weight, device=logits.device, dtype=logits.dtype)
    onset_pw  = torch.tensor(onset_pos_weight,  device=logits.device, dtype=logits.dtype)

    loss_active = F.binary_cross_entropy_with_logits(
        active_logits,
        active_target,
        pos_weight=active_pw,
    )

    loss_onset = F.binary_cross_entropy_with_logits(
        onset_logits,
        onset_target,
        pos_weight=onset_pw,
    )

    loss = loss_active + onset_loss_weight * loss_onset

    return loss, {
        "active": loss_active.detach().item(),
        "onset": loss_onset.detach().item(),
        "total": loss.detach().item(),
    }
 
 
# ── parameter count utility ───────────────────────────────────────────────────
 
def count_params(model: nn.Module) -> str:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"total={total/1e6:.2f}M  trainable={trainable/1e6:.2f}M"


