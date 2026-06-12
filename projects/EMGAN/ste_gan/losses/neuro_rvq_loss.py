"""NeuroRVQ Token Loss.

Computes an MSE loss between the continuous quantized latent vectors
extracted by the NeuroRVQ EMG tokenizer for the real and synthesized EMG.

The tokenizer is frozen (eval mode, no grad) and used purely as a
perceptual feature extractor. Gradients only flow through the synthesized
EMG path.

Expected input shape (STE-GAN convention): [B, T, C]
  B = batch size
  T = time samples  (e.g. 2048 @ 800 Hz)
  C = EMG channels  (e.g. 8)

The NeuroRVQ tokenizer expects:  [B, N, A, T_patch]
  N = number of channels
  A = number of temporal patches per channel
  T_patch = patch_size (200 by default)

This wrapper handles the reshape automatically.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor

# NeuroRVQ imports – the NeuroRVQ repository must be on PYTHONPATH or
# located at the project root as described in its README.
from NeuroRVQ_EMG.NeuroRVQ import NeuroRVQTokenizer
from NeuroRVQ_EMG.NeuroRVQ_modules import get_encoder_decoder_params
from inference.modules.NeuroRVQ_EMG_tokenizer_inference_modules import (
    ch_names_global,
    create_embedding_ix,
)


class NeuroRVQLoss(nn.Module):
    """MSE loss between NeuroRVQ quantized vectors of real and synthesized EMG.

    Args:
        cfg: OmegaConf DictConfig from STE-GAN. Reads the following fields
            under ``cfg.train``:
              - ``neuro_rvq_config_path``   path to NeuroRVQ_EMG_v1.yml
              - ``neuro_rvq_model_path``    path to the tokenizer checkpoint
              - ``neuro_rvq_ch_names``      list of channel name strings
              - ``chunk_size``              number of EMG samples per chunk
    """

    def __init__(self, cfg) -> None:
        super().__init__()

        config_path      = cfg.train.neuro_rvq_config_path
        model_weight_path = cfg.train.neuro_rvq_model_path
        ch_names_sample  = list(cfg.train.neuro_rvq_ch_names)
        max_time_samples = cfg.train.chunk_size

        # ------------------------------------------------------------------
        # 1. Load tokenizer architecture from YAML
        # ------------------------------------------------------------------
        with open(config_path, "r") as f:
            args = yaml.safe_load(f)

        args["n_global_electrodes"] = len(ch_names_global)
        encoder_config, decoder_config = get_encoder_decoder_params(args)

        self.patch_size: int = args["patch_size"]   # 200
        self.n_patches: int  = args["n_patches"]    # 256

        # ------------------------------------------------------------------
        # 2. Build and freeze the tokenizer
        # ------------------------------------------------------------------
        self.tokenizer = NeuroRVQTokenizer(
            encoder_config,
            decoder_config,
            n_code=args["n_code"],
            code_dim=args["code_dim"],
            decoder_out_dim=args["decoder_out_dim"],
        )
        self.tokenizer.load_state_dict(
            torch.load(model_weight_path, map_location="cpu")
        )
        self.tokenizer.eval()
        for param in self.tokenizer.parameters():
            param.requires_grad = False

        # ------------------------------------------------------------------
        # 3. Pre-compute spatial / temporal embedding indices
        #    The tokenizer requires these to know where each patch sits in
        #    space (channel) and time.
        # ------------------------------------------------------------------
        n_time = max_time_samples // self.patch_size

        ch_names_encoded = np.array(
            [c.lower().encode() for c in ch_names_sample]
        )
        temp_ix, spat_ix = create_embedding_ix(
            n_time=n_time,
            max_n_patches=self.n_patches,
            ch_names_sample=ch_names_encoded,
            ch_names_global=ch_names_global,
        )

        # Register as buffers so they move to GPU automatically with .to(device)
        self.register_buffer("temp_ix", temp_ix.int())
        self.register_buffer("spat_ix", spat_ix.int())

        self.n_time     = n_time
        self.n_channels = len(ch_names_sample)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _to_tokenizer_input(self, emg: Tensor) -> Tensor:
        """Reshape STE-GAN EMG [B, T, C] -> NeuroRVQ [B, C, A, T_patch]."""
        B, T, C = emg.shape
        n_samples = self.n_time * self.patch_size
        emg = emg[:, :n_samples, :]                                # [B, n_time*patch_size, C]
        emg = emg.permute(0, 2, 1)                                 # [B, C, n_time*patch_size]
        emg = emg.reshape(B, C, self.n_time, self.patch_size)      # [B, C, A, T_patch]
        return emg

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, emg_real: Tensor, emg_synth: Tensor) -> Tensor:
        """Compute NeuroRVQ token loss.

        Args:
            emg_real:  Real EMG,        shape [B, T, C].
            emg_synth: Synthesized EMG, shape [B, T, C].

        Returns:
            Scalar MSE loss summed across all RVQ scales.
        """
        x_real  = self._to_tokenizer_input(emg_real)
        x_synth = self._to_tokenizer_input(emg_synth)

        if False: # controls sampling before inference of NeuroRVQ, to reduce training time
          
          # extract 1 random patch from each channel (which has 10 patches)
          rng = np.random.default_rng()
          array_8d = rng.integers(low=0, high=10, size=(8)) + np.arange(0, 80, 10) # extracts 8 indices precisely
  
          x_real_sampled = x_real.reshape(x_real.shape[0], -1, self.patch_size)[:, array_8d, :]
          x_real = x_real_sampled.reshape(x_real.shape[0], x_real.shape[1], -1, self.patch_size)
          x_synth_sampled = x_synth.reshape(x_synth.shape[0], -1, self.patch_size)[:, array_8d, :]
          x_synth = x_synth_sampled.reshape(x_synth.shape[0], x_synth.shape[1], -1, self.patch_size)

        # Real path: no grad needed (frozen target)
        with torch.no_grad():
            out_real      = self.tokenizer.get_tokens(x_real, self.temp_ix, self.spat_ix)
            quantize_real = out_real["quantize"]    # list of [B, (A*C), code_dim]

        # Synth path: gradients flow back to the generator
        out_synth      = self.tokenizer.get_tokens(x_synth, self.temp_ix, self.spat_ix)
        quantize_synth = out_synth["quantize"]      # list of [B, (A*C), code_dim]

        # Sum MSE across all RVQ scales
        loss = sum(
            F.mse_loss(q_s, q_r)
            for q_r, q_s in zip(quantize_real, quantize_synth)
        )
        return loss
