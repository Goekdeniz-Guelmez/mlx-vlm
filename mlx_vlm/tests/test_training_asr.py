"""ASR next-token and CTC training over the shared engine with tiny models."""

import itertools
import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

from mlx_vlm.train import parse_args
from mlx_vlm.trainer import CTCTrainConfig, SFTTrainConfig, Trainer
from mlx_vlm.trainer.asr.models import ASRBridge, get_bridge
from mlx_vlm.trainer.datasets.loading import RawSplit
from mlx_vlm.trainer.losses.ctc import ctc_loss

VOCAB, MELS = 6, 4


class TinyDecoder(nn.Module):
    """Autoregressive ASR: named audio input, logits per token."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, 8)
        self.audio = nn.Linear(MELS, 8)
        self.head = nn.Linear(8, VOCAB)

    def __call__(self, input_ids, input_features=None):
        hidden = self.embedding(input_ids)
        return self.head(hidden + self.audio(input_features).mean(axis=1)[:, None])


class TinyEncoder(nn.Module):
    """Frame-synchronous ASR whose call is not a logits function."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(MELS, VOCAB)

    def logits(self, mel):
        return self.layer(mel)

    def asr_bridge(self, algorithm, config):
        return ASRBridge(
            process=lambda row: {
                "mel": features(row["audio"]),
                "input_lengths": row["frames"],
                "labels": row["labels"],
            },
            forward=lambda model, mel: model.logits(mel),
            blank_id=VOCAB - 1,
        )


def features(path, frames=None):
    seed = int(path.rsplit("/", 1)[-1].split(".")[0])
    frames = frames or 6 + seed
    return np.random.default_rng(seed).normal(size=(frames, MELS)).astype(np.float32)


def brute_force_ctc(log_probs, target, blank):
    """Sum every frame alignment that collapses to the target."""
    total = 0.0
    for path in itertools.product(range(log_probs.shape[1]), repeat=len(log_probs)):
        collapsed = [k for k, _ in itertools.groupby(path) if k != blank]
        if collapsed == target:
            total += math.exp(sum(log_probs[t, k] for t, k in enumerate(path)))
    return -math.log(total)


def test_ctc_loss_matches_alignment_enumeration_and_masks_padding():
    logits = mx.random.normal((2, 5, 3), key=mx.random.key(0))
    log_probs = np.array(nn.log_softmax(logits, axis=-1))
    expected = [
        brute_force_ctc(log_probs[0], [1, 1], blank=0),
        brute_force_ctc(log_probs[1, :3], [2], blank=0),
    ]
    loss, metrics = ctc_loss(
        logits, mx.array([[1, 1], [2, -100]]), mx.array([5, 3]), blank_id=0
    )
    assert loss.item() == pytest.approx(sum(expected) / 3, rel=1e-5)
    assert metrics["weight"].item() == 3 and metrics["num_frames"].item() == 8
    gradient = mx.grad(lambda x: ctc_loss(x, mx.array([[1, 1], [2, -100]]))[0])(logits)
    assert mx.isfinite(gradient).all().item()


def test_ctc_loss_skips_targets_longer_than_their_frames():
    logits = mx.zeros((2, 2, 3))
    loss, metrics = ctc_loss(logits, mx.array([[1, 1, 2], [1, -100, -100]]))
    assert metrics["weight"].item() == 1
    assert loss.item() == pytest.approx(
        brute_force_ctc(np.full((2, 3), -math.log(3)), [1], blank=0), rel=1e-5
    )


def test_asr_sft_trains_transcript_tokens_from_speech_rows(tmp_path):
    seen = []

    def process(row):
        seen.append(row["audio"])
        return {
            "input_ids": np.array([0, 5, *row["ids"]]),
            "completion_mask": np.array([0, 0] + [1] * len(row["ids"])),
            "input_features": features(row["audio"], frames=8),
        }

    rows = RawSplit(
        [{"audio": f"audio/{i}.wav", "ids": [1, 2, 3][: 2 + i % 2]} for i in range(4)],
        tmp_path,
    )
    model = TinyDecoder()
    trainer = Trainer(
        model,
        SFTTrainConfig(
            batch_size=2,
            iters=30,
            steps_per_report=30,
            steps_per_eval=30,
            adapter_file=str(tmp_path / "adapters.safetensors"),
        ),
        task="asr",
        train_dataset=rows,
        eval_dataset=rows,
        processing_class=process,
        optimizer=optim.Adam(learning_rate=0.05),
    )
    before = trainer.evaluate()
    metrics = trainer.train()
    assert metrics["val_loss"] < before["val_loss"]
    # Only transcript tokens are supervised; audio paths resolve to the dataset root.
    assert before["val_num_tokens"] == 10
    assert seen[0] == str(tmp_path / "audio/0.wav")


def test_asr_ctc_trains_through_a_model_bridge(tmp_path):
    rows = [
        {"audio": f"{i}.wav", "frames": 6 + i, "labels": [1, 2, 3][: 1 + i % 3]}
        for i in range(4)
    ]
    model = TinyEncoder()
    trainer = Trainer(
        model,
        CTCTrainConfig(
            batch_size=2,
            iters=40,
            steps_per_report=40,
            steps_per_eval=40,
            adapter_file=str(tmp_path / "adapters.safetensors"),
        ),
        task="asr",
        train_dataset=rows,
        eval_dataset=rows,
        optimizer=optim.Adam(learning_rate=0.05),
    )
    batch = next(trainer.task.batches(trainer.eval_dataset))
    assert batch["mel"].shape == (2, 7, MELS)
    assert batch["labels"].tolist() == [[1, -100], [1, 2]]
    before = trainer.evaluate()
    metrics = trainer.train()
    assert metrics["val_loss"] < before["val_loss"]
    assert before["val_num_frames"] == 6 + 7 + 8 + 9


def test_models_without_a_bridge_need_a_processor():
    assert get_bridge(TinyDecoder(), "sft") == ASRBridge()
    with pytest.raises(ValueError, match="processing_class"):
        Trainer(TinyDecoder(), task="asr", train_dataset=[{"audio": "a.wav"}])


def test_cli_selects_asr_algorithms():
    args, settings = parse_args(
        ["--model", "unused", "--task", "asr", "--algorithm", "ctc", "--blank-id", "7"]
    )
    assert isinstance(settings, CTCTrainConfig) and settings.blank_id == 7
    _, settings = parse_args(["--model", "unused", "--task", "asr"])
    assert isinstance(settings, SFTTrainConfig)
