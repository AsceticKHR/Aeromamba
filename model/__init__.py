"""
Public API for the AeroMamba model package.
"""

from .uav_mamba_vla   import AeroMambaVLA, MAMBA_PRESETS
from .projector       import MLPProjector
from .action_head     import UAVActionHead, UAVActionChunkHead, aero_action_loss, TemporalEnsemble
from .vision          import (
    VisionEncoder,
    DinoSigLIPEncoder,
    DinoSigLIPTransform,
    build_vision_encoder,
    SINGLE_ENCODERS,
    DINOSIGLIP_ENCODERS,
)
from .proprio_encoder import ProprioEncoder

__all__ = [
    # Main model
    "AeroMambaVLA",
    "MAMBA_PRESETS",
    # Vision
    "VisionEncoder",
    "DinoSigLIPEncoder",
    "DinoSigLIPTransform",
    "build_vision_encoder",
    "SINGLE_ENCODERS",
    "DINOSIGLIP_ENCODERS",
    # Other sub-modules
    "MLPProjector",
    "UAVActionHead",
    "UAVActionChunkHead",
    "aero_action_loss",
    "TemporalEnsemble",
    "ProprioEncoder",
]
