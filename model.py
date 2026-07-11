"""
model.py
--------
Transformer that consumes a sequence of observed frames (t = 0 .. T_obs-1)
and predicts the (x, y) location of the dot at each of the next T_future
frames, plus (optionally) estimates of p and k from the observed sequence.

Architecture (per the project spec)
------------------------------------
Each frame is d x d and gets FLATTENED into d*d scalar "pixel tokens".
All observed frames are concatenated into one long sequence of pixel
tokens: total length = T_obs * d * d.

Because every pixel token needs to know both:
    (a) WHERE in the frame it is (which of the d*d grid cells), and
    (b) WHICH frame (timestep) it came from,
we add TWO positional encodings to every token:
    - pixel positional encoding: one embedding per grid cell (0 .. d*d-1)
    - frame positional encoding: one embedding per timestep (0 .. T-1)

To predict the future, we append T_future "query" tokens to the sequence.
Each query token has no pixel content (the future is unobserved) -- it's
just a learned "query" embedding plus the frame positional encoding for
the future timestep it's asking about. The transformer attends over the
whole sequence (observed pixels + queries) and we read the (x, y)
prediction off of each query token's output representation.

p / k estimation heads read off of a pooled (mean) representation of the
observed pixel tokens.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class DotTransformer(nn.Module):
    def __init__(
        self,
        d: int,                 # grid side length (grid is d x d)
        d_model: int = 64,      # transformer embedding dimension
        n_heads: int = 4,
        n_layers: int = 1,
        max_frames: int = 200,  # must cover T_obs + T_future
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        predict_pk: bool = True,
    ):
        super().__init__()
        self.d = d
        self.n_cells = d * d
        self.d_model = d_model
        self.predict_pk = predict_pk

        # pixel value (+1/-1) -> d_model
        self.value_proj = nn.Linear(1, d_model)

        # two positional encodings
        self.pixel_pos_emb = nn.Embedding(self.n_cells, d_model)
        self.frame_pos_emb = nn.Embedding(max_frames, d_model)

        # Static index buffers (move with .to(device)/.cuda(), never rebuilt
        # per forward call). pixel_ids doesn't depend on the batch or T, so
        # it's precomputed once; frame ids do depend on T so we still slice
        # arange for those, but from a cheap 1D buffer instead of building
        # a fresh tensor+expand every call.
        self.register_buffer("_pixel_ids", torch.arange(self.n_cells), persistent=False)
        self.register_buffer("_frame_ids_all", torch.arange(max_frames), persistent=False)

        # learned "query" embedding used for every future-frame query token
        self.future_query_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # coordinate heads: classify which of the d rows / d cols the dot is in
        self.row_head = nn.Linear(d_model, d)
        self.col_head = nn.Linear(d_model, d)

        if predict_pk:
            self.p_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
            )
            self.k_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
            )

    def forward(self, frames_obs: torch.Tensor, t_future: int):
        """
        frames_obs: (B, T_obs, d, d) float tensor with values in {-1, +1}
        t_future:   number of future frames to predict

        Returns dict with:
            row_logits: (B, t_future, d)
            col_logits: (B, t_future, d)
            p_pred:     (B,)  (only if predict_pk)
            k_pred:     (B,)  (only if predict_pk)
        """
        B, T, d, _ = frames_obs.shape
        n_cells = d * d

        pixels = frames_obs.view(B, T, n_cells, 1)  # (B, T, n_cells, 1)
        tok = self.value_proj(pixels)                # (B, T, n_cells, d_model)

        # Broadcast (no .expand()/materialization needed): embedding lookup
        # on a (1,1,n_cells) index already broadcasts against (B,T,n_cells,d_model)
        # when added, so skip the explicit .expand() copies.
        pixel_ids = self._pixel_ids.view(1, 1, n_cells)
        tok = tok + self.pixel_pos_emb(pixel_ids)

        frame_ids_obs = self._frame_ids_all[:T].view(1, T, 1)
        tok = tok + self.frame_pos_emb(frame_ids_obs)

        tok = tok.reshape(B, T * n_cells, self.d_model)  # (B, T*n_cells, d_model)

        # future query tokens
        fq_frame_ids = self._frame_ids_all[T:T + t_future].view(1, t_future)
        fq = self.future_query_emb.expand(B, t_future, self.d_model)
        fq = fq + self.frame_pos_emb(fq_frame_ids)

        seq = torch.cat([tok, fq], dim=1)  # (B, T*n_cells + t_future, d_model)

        out = self.encoder(seq)

        future_out = out[:, -t_future:, :]          # (B, t_future, d_model)
        obs_out = out[:, : T * n_cells, :].mean(dim=1)  # (B, d_model)

        result = {
            "row_logits": self.row_head(future_out),
            "col_logits": self.col_head(future_out),
        }
        if self.predict_pk:
            result["p_pred"] = torch.sigmoid(self.p_head(obs_out)).squeeze(-1)
            result["k_pred"] = self.k_head(obs_out).squeeze(-1)
        return result

    def num_tokens(self, t_obs: int, t_future: int) -> int:
        """Helper to sanity-check sequence length / compute cost before training."""
        return t_obs * self.n_cells + t_future
