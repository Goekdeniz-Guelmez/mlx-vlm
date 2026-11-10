"""CTC training for frame-synchronous (streaming-capable) ASR models.

Examples hold ``labels`` (target token ids), the model's audio inputs under
its own forward argument names, and optionally ``input_lengths``: the
unpadded audio length in any unit, used to ignore padded output frames.
"""

from functools import partial

import mlx.core as mx
import numpy as np

from mlx_vlm.trainer.asr.common import ASRDataset, named_forward
from mlx_vlm.trainer.asr.models import get_bridge
from mlx_vlm.trainer.common.task import TrainingTask
from mlx_vlm.trainer.common.utils import (
    enable_model_gradient_checkpointing,
    iterate_batch_indices,
    pad_arrays,
    save_trainable_weights,
)
from mlx_vlm.trainer.datasets.loading import CacheDataset
from mlx_vlm.trainer.losses.ctc import IGNORE_INDEX, ctc_loss
from mlx_vlm.trainer.vlm.cross_entropy.runtime import (
    _collate_feature,
    _logits_from_output,
)

_SUPERVISION_FIELDS = frozenset({"labels", "input_lengths"})


def _labels(sample) -> np.ndarray:
    return np.asarray(sample["labels"], dtype=np.int32).reshape(-1)


def collate_ctc_batch(samples) -> dict:
    """Zero-pad model inputs to the longest example and pad labels to ignore."""
    batch = {
        key: _collate_feature([sample[key] for sample in samples], key)
        for key in samples[0]
        if key not in _SUPERVISION_FIELDS
    }
    batch["labels"] = mx.array(
        pad_arrays([_labels(sample) for sample in samples], padding_value=IGNORE_INDEX)
    )
    if "input_lengths" in samples[0]:
        batch["input_lengths"] = mx.array(
            [float(np.asarray(sample["input_lengths"])) for sample in samples]
        )
    return batch


def iterate_ctc_batches(dataset, batch_size, *, train=False, seed=None):
    """Length-grouped CTC batches, sharded across distributed ranks."""
    world = mx.distributed.init()

    def length(index):
        sample = dataset[index]
        return float(np.asarray(sample.get("input_lengths", len(_labels(sample)))))

    for indices in iterate_batch_indices(
        len(dataset),
        batch_size,
        length_key=length,
        train=train,
        rank=world.rank(),
        world_size=world.size(),
        seed=seed,
    ):
        yield collate_ctc_batch([dataset[index] for index in indices])


def asr_ctc_loss(model, batch, *, call=None, blank_id=0):
    """Run the model on its named inputs and score frame logits with CTC."""
    call = call or named_forward(model)
    logits = _logits_from_output(
        call({k: v for k, v in batch.items() if k not in _SUPERVISION_FIELDS})
    )
    frames = batch.get("input_lengths")
    if frames is not None:
        # The batch is padded to its longest example, so the valid share of
        # output frames equals the valid share of input, whatever the stride.
        frames = mx.ceil(frames / frames.max() * logits.shape[1])
    return ctc_loss(
        logits.astype(mx.float32), batch["labels"], frames, blank_id=blank_id
    )


def make_task(model, args, processing_class=None, dataset_config=None):
    """Bind ASR examples to the CTC objective."""
    bridge = get_bridge(model, "ctc", dataset_config)
    blank_id = args.blank_id if args.blank_id is not None else bridge.blank_id or 0
    return TrainingTask(
        loss=partial(
            asr_ctc_loss, call=named_forward(model, bridge.forward), blank_id=blank_id
        ),
        batches=partial(iterate_ctc_batches, batch_size=args.batch_size),
        prepare_dataset=lambda rows: CacheDataset(
            ASRDataset(rows, processing_class or bridge.process),
            max_size=args.cache_size,
            process=False,
        ),
        checkpoint=enable_model_gradient_checkpointing,
        save_weights=save_trainable_weights,
        unit="tokens",
    )
