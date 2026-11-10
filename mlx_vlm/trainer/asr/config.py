"""Settings for ASR objectives; next-token training reuses ``SFTTrainConfig``."""

from dataclasses import dataclass, field

from mlx_vlm.trainer.common.config import BaseTrainConfig


@dataclass
class CTCTrainConfig(BaseTrainConfig):
    """Connectionist Temporal Classification settings."""

    blank_id: int | None = field(
        default=None,
        metadata={"help": "CTC blank class; None uses the model's blank, else 0."},
    )

    def validate(self) -> None:
        super().validate()
        if self.blank_id is not None and self.blank_id < 0:
            raise ValueError("blank_id must be nonnegative.")
