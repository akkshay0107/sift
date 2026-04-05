from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ClapModel, ClapProcessor


@dataclass
class AudioSegment:
    data: np.ndarray
    sample_rate: int
    t_start: float
    t_end: float
    source_id: str
    transcript: Optional[str] = None
    text_embedding: Optional[torch.Tensor] = None
    audio_embedding: Optional[torch.Tensor] = None


CLAP_AUDIO_DIM = 512
QWEN_EMBED_DIM = 2048
_DEFAULT_CLAP_MODEL = "laion/clap-htsat-unfused"


class ProjectionHead(nn.Module):
    """
    Two-layer MLP that projects CLAP audio embeddings into the Qwen anchor space.

    Parameters
    ----------
    in_dim : int
        Dimensionality of the CLAP audio embedding (default 512).
    hidden_dim : int
        Hidden layer width (default 1024).
    out_dim : int
        Target dimensionality, i.e. the Qwen anchor space (default 2048).
    dropout : float
        Dropout probability applied after GELU (default 0.0).
    """

    def __init__(
        self,
        in_dim: int = CLAP_AUDIO_DIM,
        hidden_dim: int = 2048,
        out_dim: int = QWEN_EMBED_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape [B, in_dim]

        Returns
        -------
        torch.Tensor, shape [B, out_dim]
        """
        return self.net(x)


class AudioEmbedder:
    """
    Audio embedding model using CLAP + a learned projection head.

    Produces L2-normalized float32 embeddings of shape [N, 2048] on CPU,
    matching the output contract of QwenEmbedder.

    Parameters
    ----------
    clap_model_id : str
        HuggingFace model id or local path for the CLAP model.
        Defaults to 'laion/clap-htsat-unfused'.
    projection_path : str | None
        Path to a saved ProjectionHead state-dict (.pt file).
        If None, the projection head is randomly initialized (useful during
        training or when you only need CLAP features).
    device : str | torch.device | None
        Compute device.  Defaults to CUDA if available, else CPU.
    dtype : torch.dtype
        Weight dtype for CLAP.  float32 is safest for CPU; bfloat16 works
        on CUDA.
    """

    CLAP_SAMPLE_RATE: int = 48_000

    def __init__(
        self,
        clap_model_id: str = _DEFAULT_CLAP_MODEL,
        projection_path: Optional[str] = None,
        device: Optional[str | torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.dtype = dtype

        # -- CLAP backbone (audio encoder only) --
        self._processor = ClapProcessor.from_pretrained(clap_model_id)
        clap: ClapModel = ClapModel.from_pretrained(clap_model_id, torch_dtype=dtype)
        # We only need the audio tower; discard the text tower to save memory.
        self._audio_model = clap.audio_model.to(self.device).eval()
        self._audio_projection = clap.audio_projection.to(self.device).eval()

        # -- Projection head (512 → 2048) --
        self._projection = ProjectionHead(dropout=0.2).to(self.device)
        if projection_path is not None:
            state = torch.load(projection_path, map_location=self.device)
            # Support loading both raw state_dicts and nested checkpoints from train_loop.py
            if "proj_head_state_dict" in state:
                self._projection.load_state_dict(state["proj_head_state_dict"])
            else:
                self._projection.load_state_dict(state)
        self._projection.eval()

    def embed(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> torch.Tensor:
        """
        Embed a single audio clip.

        Parameters
        ----------
        audio : np.ndarray
            1-D (mono) float32 waveform array.
        sample_rate : int
            Sample rate of *audio* in Hz.

        Returns
        -------
        torch.Tensor, shape [1, 2048], dtype float32, L2-normalized, on CPU.
        """
        return self.embed_batch([(audio, sample_rate)])

    def embed_batch(
        self,
        clips: list[tuple[np.ndarray, int]],
    ) -> torch.Tensor:
        """
        Embed a batch of audio clips in a single forward pass.

        Parameters
        ----------
        clips : list of (waveform: np.ndarray, sample_rate: int)
            Each waveform should be 1-D float32.  Clips with different
            lengths are zero-padded by the CLAP processor.

        Returns
        -------
        torch.Tensor, shape [N, 2048], dtype float32, row-wise L2-normalized, on CPU.
        """
        if not clips:
            raise ValueError("clips must be a non-empty list")

        # Resample every clip to CLAP's required 48 kHz.
        audios = [self._resample(c[0], c[1]) for c in clips]

        inputs = self._processor(
            audio=audios,
            sampling_rate=self.CLAP_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs["input_features"].to(self.device, dtype=self.dtype)

        with torch.no_grad():
            audio_outputs = self._audio_model(input_features=input_features)
            pooled = audio_outputs.pooler_output  # [B, hidden_dim] e.g. 768
            clap_embed = self._audio_projection(pooled)  # [B, 512]
            clap_embed = F.normalize(clap_embed, p=2, dim=-1)
            projected = self._projection(clap_embed.float())  # [B, 2048]
            projected = F.normalize(projected, p=2, dim=-1)

        return projected.cpu()

    def embed_segment(self, segment: AudioSegment) -> torch.Tensor:
        """
        Embed an AudioSegment and store the result in segment.audio_embedding.

        Returns
        -------
        torch.Tensor, shape [1, 2048], float32, CPU.
        """
        emb = self.embed(segment.data, segment.sample_rate)
        segment.audio_embedding = emb
        return emb

    def _resample(self, audio: np.ndarray, src_sr: int) -> np.ndarray:
        """Resample *audio* from *src_sr* to CLAP_SAMPLE_RATE (48 kHz) if needed."""
        if src_sr == self.CLAP_SAMPLE_RATE:
            return audio
        target_length = int(len(audio) * self.CLAP_SAMPLE_RATE / src_sr)
        src_indices = np.linspace(0, len(audio) - 1, target_length)
        return np.interp(src_indices, np.arange(len(audio)), audio).astype(np.float32)

    def projection_head(self) -> ProjectionHead:
        """Return the projection head (e.g. to pass its parameters to an optimizer)."""
        return self._projection
