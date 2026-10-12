"""Load a trusted Lightning checkpoint with its saved Hydra configuration."""
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def load_model(ckpt_path, device='cpu', config_path=None, parameter_config=None):
    ckpt_path = Path(ckpt_path).resolve()
    config_path = Path(config_path) if config_path else ckpt_path.parent.parent / '.hydra/config.yaml'
    if not config_path.is_file():
        raise FileNotFoundError(f'Missing model configuration: {config_path}. Supply --config.')
    cfg = OmegaConf.load(config_path)
    if parameter_config is None:
        saved = config_path.parent / 'para_info.yaml'
        if saved.is_file():
            parameter_config = saved
        else:
            original = Path(str(cfg.data.paras_cfg_path))
            if original.is_file():
                parameter_config = original
            else:
                raise FileNotFoundError('Missing parameter normalization. Supply --parameter-config.')
    parameter_config = Path(parameter_config).resolve()
    cfg.data.paras_cfg_path = str(parameter_config)
    cfg.model.paras_cfg_path = str(parameter_config)
    cfg.model.load_pretrained_ckpt = None
    model = hydra.utils.instantiate(cfg.model)
    # Lightning checkpoints may contain Python objects. Load files you trust only.
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device).eval()
    return model, cfg, OmegaConf.load(parameter_config)
