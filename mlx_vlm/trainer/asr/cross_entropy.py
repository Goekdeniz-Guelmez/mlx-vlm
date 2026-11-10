"""Next-token cross-entropy for autoregressive ASR models.

Examples hold unshifted ``input_ids`` (prompt, audio placeholders, transcript),
a ``completion_mask`` marking the supervised transcript tokens, and the
model's audio inputs under its forward argument names. The objective,
collation, and batching are the VLM ones; only the model call differs.
"""

from dataclasses import replace
from functools import partial

from mlx_vlm.trainer.asr.common import ASRDataset, named_forward
from mlx_vlm.trainer.asr.models import get_bridge
from mlx_vlm.trainer.vlm.cross_entropy import runtime


def forward_asr(model, input_ids, attention_mask, batch, *, call, **kwargs):
    """Pass shifted tokens and audio fields; padding is right of causal context."""
    fields = {
        key: value for key, value in batch.items() if key not in runtime._TEXT_FIELDS
    }
    return call({**fields, "input_ids": input_ids, **kwargs})


def make_task(model, args, processing_class=None, dataset_config=None):
    """Bind ASR examples to the shared supervised cross-entropy task."""
    bridge = get_bridge(model, "sft", dataset_config)
    forward = partial(forward_asr, call=named_forward(model, bridge.forward))
    return runtime.make_task(
        model,
        # Prompts and audio placeholders are never transcript targets.
        replace(args, train_on_completions=True, assistant_id=None),
        loss_fn=partial(runtime.vlm_sft_loss, forward=forward),
        dataset_factory=lambda rows: ASRDataset(
            rows, processing_class or bridge.process
        ),
    )
