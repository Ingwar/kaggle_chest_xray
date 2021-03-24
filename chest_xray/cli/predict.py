from pathlib import Path

from pytorch_lightning import Trainer

from .args import setup_common_cli_parser
from .hydra import parse_hydra_config
from ..data.datamodule import XRayDataModule
from ..model.experiment import Experiment
from ..prediction import generate_submission_file, parse_predictions
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
    predictions = trainer.predict(experiment, datamodule=data)
    parsed_predictions = parse_predictions(predictions, data.predict_dataset.image_ids)
    generate_submission_file(config.submission.file, parsed_predictions, config.evaluation.confidence_threshold)


if __name__ == '__main__':
    run()

