"""Bridges from loaded ASR models to the shared speech row format.

A bridge turns ``{"audio": path, "text": str}`` rows into examples named after
the forward's arguments, and supplies a logits forward when the model's own
call is an inference interface. Models that define
``asr_bridge(algorithm, config)`` or already take named fields need no entry.
"""

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass
class ASRBridge:
    process: Callable | None = None  # row -> example
    forward: Callable | None = None  # (model, **fields) -> logits
    blank_id: int | None = None  # CTC blank class


def _load_audio(row):
    from mlx_audio.stt.utils import load_audio

    return load_audio(row["audio"])


def _language(row, config) -> str:
    return row.get("language") or getattr(config, "language", None) or "en"


def _sft_example(prompt, transcript, **inputs):
    """Join prompt and transcript; only the transcript is supervised."""
    prompt, transcript = (
        np.asarray(ids, dtype=np.int32) for ids in (prompt, transcript)
    )
    return {
        "input_ids": np.concatenate([prompt, transcript]),
        "completion_mask": np.concatenate(
            [np.zeros_like(prompt), np.ones_like(transcript)]
        ),
        **inputs,
    }


def _qwen3_asr_sft(model, config):
    from mlx_audio.stt.models.whisper.tokenizer import LANGUAGES

    tokenizer = model._tokenizer

    def process(row):
        language = _language(row, config)
        features, mask, count = model._preprocess_audio(row["audio"])
        prompt = model._build_prompt(count, LANGUAGES.get(language, language).title())
        transcript = tokenizer.encode(row["text"], add_special_tokens=False)
        return _sft_example(
            np.asarray(prompt)[0],
            [*transcript, tokenizer.eos_token_id],
            input_features=features,
            feature_attention_mask=mask,
        )

    return ASRBridge(process)


def _whisper_sft(model, config):
    from mlx_audio.stt.models.whisper.audio import (
        N_FRAMES,
        log_mel_spectrogram,
        pad_or_trim,
    )

    def process(row):
        tokenizer = model.get_tokenizer(language=_language(row, config))
        mel = log_mel_spectrogram(_load_audio(row), n_mels=model.dims.n_mels)
        if mel.shape[0] > N_FRAMES:
            raise ValueError("Split Whisper training audio into clips of at most 30 s.")
        return _sft_example(
            tokenizer.sot_sequence_including_notimestamps,
            [*tokenizer.encode(" " + row["text"].strip()), tokenizer.eot],
            mel=pad_or_trim(mel, N_FRAMES, axis=0),
        )

    def forward(model, input_ids, mel):
        return model(mel.astype(model.encoder.conv1.weight.dtype), input_ids)

    return ASRBridge(process, forward)


def _parakeet_ctc(model, config):
    head = getattr(model, "ctc_decoder", None) or model.decoder
    if type(head).__name__ != "ConvASRDecoder":
        raise ValueError("This Parakeet checkpoint has no CTC head.")
    pieces = {piece: index for index, piece in enumerate(model.vocabulary)}
    cased = any(piece != piece.lower() for piece in pieces)

    def encode(text):
        """BPE-encode by merging pairs in vocabulary order, as SentencePiece does."""
        symbols = list("▁" + "▁".join((text if cased else text.lower()).split()))
        while True:
            rank, index = min(
                (
                    (pieces.get(left + right, len(pieces)), index)
                    for index, (left, right) in enumerate(zip(symbols, symbols[1:]))
                ),
                default=(len(pieces), 0),
            )
            if rank == len(pieces):
                # Characters the vocabulary cannot spell are dropped.
                return [pieces[symbol] for symbol in symbols if symbol in pieces]
            symbols[index : index + 2] = [symbols[index] + symbols[index + 1]]

    def process(row):
        mel = model._log_mel_spectrogram(_load_audio(row))[0]
        return {
            "mel": mel,
            "lengths": mel.shape[0],
            "input_lengths": mel.shape[0],
            "labels": encode(row["text"]),
        }

    def forward(model, mel, lengths=None):
        features, _ = model.encoder(mel, lengths)
        return head(features)

    return ASRBridge(process, forward, blank_id=len(model.vocabulary))


# Keyed by the model's package name (``...models.<family>...``) and algorithm.
BRIDGES = {
    ("qwen3_asr", "sft"): _qwen3_asr_sft,
    ("whisper", "sft"): _whisper_sft,
    ("parakeet", "ctc"): _parakeet_ctc,
}


def get_bridge(model, algorithm: str, config=None) -> ASRBridge:
    """Return the model's own bridge, a registered one, or a pass-through."""
    if hasattr(model, "asr_bridge"):
        return model.asr_bridge(algorithm, config)
    parts = type(model).__module__.split(".")
    family = parts[parts.index("models") + 1] if "models" in parts[:-1] else None
    factory = BRIDGES.get((family, algorithm))
    return factory(model, config) if factory else ASRBridge()
