# -*- coding: utf-8 -*-
"""
Dataset Generators for Neuro-Diffusion Inference.

This package provides tools for generating synthetic neural network datasets
consisting of weight matrices and their corresponding activity traces.

Main components:
    - sample_weight_matrix: Generate random synaptic weight matrices
    - generate_activity_trials: Run multiple simulation trials
    - build_dataset: Complete dataset generation pipeline
"""

from data.generators.generate_weights import (
    sample_weight_matrix,
    sample_weight_matrix_batch,
    get_spectral_radius,
)
from data.generators.simulate_activity import (
    generate_activity_trials,
    generate_activity_trials_batched,
    compute_trial_summary,
)
from data.generators.build_dataset import (
    build_dataset,
    load_network,
    load_metadata,
    list_networks,
)

__all__ = [
    # Weight generation
    "sample_weight_matrix",
    "sample_weight_matrix_batch",
    "get_spectral_radius",
    # Activity simulation
    "generate_activity_trials",
    "generate_activity_trials_batched",
    "compute_trial_summary",
    # Dataset building
    "build_dataset",
    "load_network",
    "load_metadata",
    "list_networks",
]



