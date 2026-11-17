"""WASB ball detector (Tarashima et al., BMVC 2023): vendored HRNet, config and weight loading.

    from pipeline import wasb
    model = wasb.load_model(wasb.fetch_pretrained(wasb.PRETRAINED_PATH), wasb.pick_device())

Three frames in (9 channels, 512x288), three ball heatmaps out (logits): model(x)[0].
"""

import os
import urllib.request

import torch
import yaml
from omegaconf import OmegaConf

from pipeline.wasb.hrnet import HRNet

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# Params
PRETRAINED_PATH = os.path.join(ROOT, "pipeline", "cache", "ball_video", "wasb_soccer_best.pth.tar")
PRETRAINED_DRIVE_ID = "1pg0MpMtKZ6ziYEr4oyfKYPOO3hjLw94l"   # wasb_soccer_best.pth.tar, MODEL_ZOO.md


def pick_device():
    """"mps" on Apple silicon (needs torch 2.9.1, see requirements-video.txt), otherwise "cpu"."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def build_hrnet():
    """The WASB soccer architecture, randomly initialised."""
    with open(os.path.join(HERE, "wasb.yaml")) as handle:
        return HRNet(OmegaConf.create(yaml.safe_load(handle)))


def load_model(path, device):
    """HRNet with weights from `path` (a state dict, or {"model_state_dict": ...}), in eval mode."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("model_state_dict", state)
    model = build_hrnet()
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    return model.eval().to(device)


def fetch_pretrained(dest):
    """Download the released soccer checkpoint to `dest` unless it is already there."""
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        url = ("https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
               .format(PRETRAINED_DRIVE_ID))
        urllib.request.urlretrieve(url, dest + ".part")
        os.replace(dest + ".part", dest)
    return dest
