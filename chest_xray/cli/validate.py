from pathlib import Path

from pytorch_lightning import Trainer

from .args import setup_common_cli_parser
from .hydra import parse_hydra_config
from ..data.datamodule import XRayDataModule
from ..model.experiment import Experiment
from ..utils import silence_pydicom_warnings


def run() -> None:
    parser = setup_common_cli_parser()
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    config = parse_hydra_config(args)

    silence_pydicom_warnings()

    trainer = Trainer.from_argparse_args(args)
    data = XRayDataModule(config.data)
    experiment = Experiment.load_from_checkpoint(args.checkpoint)
    result = trainer.test(experiment, datamodule=data)
    print(result)


if __name__ == '__main__':
    run()

