import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from backend.config import CFG, ExperimentConfig

# This harmonic stacking rearranges the cqt input so that each harmonic is stacked along the channel dimension.
# Instead of having to search the entire frequency axis to find the 2nd, 3rd, 4th, etc. harmonics of a given pitch, the model can just look at the stacked channels to find them.
# Note that it doesn't account for the fact that on a real piano, higher overtones are sharper, so the formula is not exact in practice.
# This could be improved by making the shifts learnable parameters, which is something I'll try later
class HarmonicStacking(nn.Module):
    """
    Takes CQT input (B, 1, T, F) and stacks shifted frequency views.

    Each harmonic ratio h corresponds to a shift of:
        bins_per_octave * log2(h)
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

        # Precompute the shifts for each harmonic ratio. This is a fixed mapping from harmonic ratio to frequency bin shift.
        # This eliminates the need to compute log2(h) for every forward pass, which is more efficient.
        self.shifts = [
            int(round(bins_per_octave * math.log2(h)))
            for h in harmonics
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, F)
        assert x.ndim == 4
        _, C, _, _ = x.shape
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

        # (B, 8, T, F) where 8 is the number of harmonics in the channel dimension. This allows the model to see all harmonics at once for each pitch bin.
        x = torch.cat(stacked, dim=1)

        if self.target_bins is not None:
            x = x[..., :self.target_bins]

        return x


# Generic little conv with some bn and gelu
class ConvBlock(nn.Module):
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

# Generic ffn used in the attention block
class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


# attention over the pitch axis to help the model learn relationships between pitches
class PitchAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = SDPASelfAttention(d_model, n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, P, C = x.shape
        h = x.reshape(B * T, P, C)
        h = self.attn(h)
        h = h.reshape(B, T, P, C)
        return self.norm(x + h)


# SDPA is more memory efficient than standard mha which reduces training time quite a bit
class SDPASelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Bseq, L, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(Bseq, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  

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

# Positional embedding for our pitch attn. Does not use time positional embedding because it is way too computationally expensive and is not worth the train time increase. 
class PitchOnlyPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.pitch_emb = nn.Embedding(CFG.feature.n_pitches, d_model)

    def forward(self, x):
        # x: (B, T, 88, C)
        B, T, P, C = x.shape
        p_idx = torch.arange(P, device=x.device)
        p_emb = self.pitch_emb(p_idx).view(1, 1, P, C)
        return x + p_emb

# Axial attention helps the model learn relationships betwen pitches, which is important because piano music uses chords heavily
# If C and E are being played, the model will take a closer look at G and ignore noise at other spots like F#
# I want to try adding Time Attention as well and replacing the dilations later. We might need to use localized attention (maybe +-64 frames) since 384 is a lot
class AxialAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.pos = PitchOnlyPositionalEmbedding(d_model)
        self.pitch_attn = PitchAttention(d_model, n_heads, dropout=dropout)
        self.ff = FeedForward(d_model, expansion=2, dropout=dropout)

    def forward(self, x):
        # x: (B, C, T, 88)
        x = x.permute(0, 2, 3, 1)  # (B, T, 88, C)

        x = self.pos(x)
        x = self.pitch_attn(x)
        x = self.ff(x)

        x = x.permute(0, 3, 1, 2)  # (B, C, T, 88)
        return x


# Try a GRU here
class PitchwiseBiGRU(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError("d_model must be even for this BiGRU")

        # Each direction produces d_model // 2 features.
        # Concatenating forward and backward gives d_model again.
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, P)
        B, C, T, P = x.shape

        # Give each pitch its own temporal sequence.
        h = x.permute(0, 3, 2, 1).contiguous()
        # h: (B, P, T, C)

        h = h.reshape(B * P, T, C)
        # h: (B*P, T, C)

        temporal, _ = self.gru(h)
        # temporal: (B*P, T, C)

        # Residual refinement rather than complete replacement.
        h = self.norm(h + self.dropout(temporal))

        h = h.reshape(B, P, T, C)
        h = h.permute(0, 3, 2, 1).contiguous()

        # (B, C, T, P)
        return h
# Separate conv block used in the detection heads. It is stronger than a normal conv block because it uses depthwise separable convolutions
class SepConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=(7, 7), padding=(3, 3)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)

# A cheaper way to learn temporal relationships. Attention over time is extremely expensive when we have a large frame count so this is more practical
class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(5, 1),
                padding=(2 * dilation, 0),
                dilation=(dilation, 1),
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)

# Main model
class PianoTranscriber(nn.Module):
    def __init__(
        self,
        n_bins: int | None = None,
        d_model: int | None = None,
        cfg: ExperimentConfig = CFG,
    ):
        super().__init__()
        n_bins = cfg.feature.input_bins if n_bins is None else n_bins
        d_model = cfg.model.d_model if d_model is None else d_model


        self.hstack = HarmonicStacking(
            bins_per_octave=cfg.feature.cqt_bins_per_octave,
            harmonics=(0.5, 1, 2, 3, 4, 5, 6, 7),
            target_bins=n_bins,
        )

        # Convert 8 channel harmonic stack to 32 and then 64 channels to learn something meaningful here
        self.trunk_pre = nn.Sequential(
            ConvBlock(8, 32, kernel=5),
            ConvBlock(32, 64, kernel=3),
            # I may consider adding a 1x1 conv here as well
            # Or maybe start with a 1x1 and use a residual to learn harmonic relationships
        )

        # Reduce the frequency axis from 264 bins to 88 bins using a local convolution with stride over frequency. This is important because the model doesn't need to see all 264 bins, and reducing the frequency axis helps the model train faster. 
        # There's 88 keys on the piano so we collapse these CQT bins into 88 keys
        # Depthwise Separable is so much cheaper than a full conv here
        self.freq_reduce = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 7), stride=(1, 3), padding=(0, 3), groups=64, bias=False), 
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        # Increase channel dim to d_model for attention block
        self.trunk_post = nn.Sequential(
            ConvBlock(64, d_model, kernel=3),
        )
        
        self.attn = AxialAttentionBlock(d_model, n_heads=cfg.model.n_heads, dropout=cfg.model.dropout)
        self.gru = PitchwiseBiGRU(
            d_model=d_model,
            num_layers=1,
            dropout=cfg.model.dropout,
        )

        # Pool across pitch after the BiGRU and predict one sustain-pedal state per frame.
        self.pedal_head = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Conv1d(d_model // 2, 1, kernel_size=1),
        )

        # This conv creates an initial frame activity prediction.
        # It'll end up with a (B, 1, T, 88) shape. The channel dim gets squished to 1 because at each pitch we predict if its active
        self.note_conv = nn.Sequential(
            SepConvBlock(d_model, kernel_size=(7, 7), padding=(3, 3)),
            nn.Conv2d(d_model, 1, kernel_size=(7, 3), padding=(3, 1)), # 1 channel. 1 value. Not to be confused with the 1x1 kernel we use in the depthwise separable conv right before it.
        )

        # Simple bunch of convs for onsets and offests
        # Onsets are much easier than offests because they don't have the sus pedal interfering with them, so a lower f1 score for offsets is normal.
        self.onset_conv = nn.Sequential(
            nn.Conv2d(d_model + 1, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            SepConvBlock(d_model, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv2d(d_model, 1, kernel_size=(3, 3), padding=(1, 1)),
        )
    
        self.offset_conv = nn.Sequential(
            nn.Conv2d(d_model + 2, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            SepConvBlock(d_model, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv2d(d_model, 1, kernel_size=(3, 3), padding=(1, 1)),
        )

        # Mix channels to go back to d_model at every (T, 88) location. 
        self.frame_refine_input = nn.Sequential(
            nn.Conv2d(d_model + 4, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )
        

        # This is the actual refinement. Using everything we learned about the note, onsets, and offsets, we refine our original guess and concat them. 
        self.frame_delta = nn.Conv2d(
            d_model,
            1,
            kernel_size=1,
        )

        # Zero-init is needed so that the model makes meaningful refinements to reduce loss
        # If it was random-init, then noisy and meaningless corrections would be made from step 1, making the output worse before layers can ever learn anything useful. 
        nn.init.zeros_(self.frame_delta.weight)
        nn.init.zeros_(self.frame_delta.bias)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
    
        x = self.hstack(x)
        h = self.trunk_pre(x)
        h = self.freq_reduce(h)
        h = self.trunk_post(h)
        h = self.attn(h)
        h = self.gru(h)
    
        pedal_features = h.mean(dim=3)  # (B, d_model, T)
        pedal_logit = self.pedal_head(pedal_features)  # (B, 1, T)
        pedal_prob = torch.sigmoid(pedal_logit)
        pedal_map = pedal_prob.unsqueeze(-1).expand(-1, -1, -1, h.shape[-1])

        # Initial activity prediction. It's a guess that gets refined later. This is important because the model needs to have a rough idea of where notes are before it can predict onsets and offsets.
        frame_seed = self.note_conv(h)

        # The onset head intentionally receives no pedal context.
        onset_input = torch.cat([h, frame_seed], dim=1)
        onset_logit = self.onset_conv(onset_input)

        # Offset can use the predicted global pedal state, without sending its loss
        # gradient back through the pedal head.
        offset_input = torch.cat([h, frame_seed, pedal_map.detach()], dim=1)
        offset_logit = self.offset_conv(offset_input)
    
        # detach logits so that active loss doesn't backprop through onset/offset predictions
        # they will use their own losses to learn
        event_context = torch.cat(
            [
                h,
                frame_seed,
                torch.sigmoid(onset_logit).detach(), #sigmoid to squash to a probability
                torch.sigmoid(offset_logit).detach(),#sigmoid to squash to a probability
                pedal_map.detach(),
            ],
            dim=1,
        ) # should be (B, d_model + 4, T, 88)
    
        refine = self.frame_refine_input(event_context)

        # Refine our guess
        frame_logit = frame_seed + self.frame_delta(refine)
    
        logits = torch.cat(
            [frame_logit, onset_logit, offset_logit], # this EXACT concat order to match with CH_ACTIVE, CH_ONSET, CH_OFFSET
            dim=1,
        )
        # Permute to (B, T, 88, 3) for loss calculation and evaluation
        return logits.permute(0, 2, 3, 1), pedal_logit