"""Training: the duration and spectral models, sized for a free Kaggle GPU."""

from .trainer import TrainConfig, train

__all__ = ["TrainConfig", "train"]
