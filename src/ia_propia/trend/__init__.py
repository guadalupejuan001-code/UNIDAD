"""Train a small trend regressor from random weights, with no pretrained model."""

from .data import DataError, load_observations, make_samples, chronological_split
from .model import ArtifactError, load_artifact, predict, train

__all__ = ["DataError", "ArtifactError", "load_observations", "make_samples", "chronological_split", "load_artifact", "predict", "train"]
