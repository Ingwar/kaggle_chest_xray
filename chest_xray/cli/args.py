from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

from pytorch_lightning import Trainer

__all__ = [
    'setup_common_cli_parser',
]


def setup_common_cli_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.formatter_class = ArgumentDefaultsHelpFormatter
    parser.add_argument('--config-dir', type=Path, default=Path('conf'))
    parser.add_argument('--config-file-name', default='config.yaml')
    parser.add_argument('--print-final-config', action='store_true')
    parser.add_argument('overrides', nargs='*')
    return parser
