import os
from argparse import Namespace
from pathlib import Path

__all__ = [
    'init_hydra',
    'parse_hydra_config',
]

from typing import cast

from hydra.core.config_store import ConfigStore
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from ..conf import PipelineConfig


def init_hydra(config_dir: Path) -> None:
    cs = ConfigStore.instance()
    cs.store(name='config', node=PipelineConfig)
    current_file = Path(__file__)
    relative_config_dir = os.path.relpath(config_dir, current_file.parent)
    initialize(relative_config_dir)


def parse_hydra_config(args: Namespace) -> PipelineConfig:
    init_hydra(args.config_dir)
    config = cast(PipelineConfig, compose(args.config_file_name, args.overrides))
    if args.print_final_config:
        print(OmegaConf.to_yaml(config, resolve=True))
    return config
