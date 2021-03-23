from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from .hydra import init_hydra, parse_hydra_config
from ..data.datamodule import XRayDataModule
from ..model import instantiate_model
from ..model.experiment import Experiment
from ..prediction import generate_submission_file, parse_predictions


def run() -> None:
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.formatter_class = ArgumentDefaultsHelpFormatter
    parser.add_argument('--config-dir', type=Path, default=Path('conf'))
    parser.add_argument('--config-file-name', default='config.yaml')
    parser.add_argument('--print-final-config', action='store_true')
    parser.add_argument('overrides', nargs='*')

    args = parser.parse_args()
    init_hydra(args.config_dir)
    config = parse_hydra_config(args)

    if args.print_final_config:
        print(OmegaConf.to_yaml(config, resolve=True))

    trainer = Trainer.from_argparse_args(args)
    data = XRayDataModule(config.data)
    model = instantiate_model(config.model.num_classes, config.model.trainable_backbone_layers)
    experiment = Experiment(model, config)
    trainer.fit(experiment, datamodule=data)
    predictions = trainer.predict(experiment, datamodule=data)
    parsed_predictions = parse_predictions(predictions, data.predict_dataset.image_ids)
    generate_submission_file(config.submission.file, parsed_predictions)


if __name__ == '__main__':
    run()
