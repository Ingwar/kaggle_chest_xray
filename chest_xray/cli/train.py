from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

from .args import setup_common_cli_parser
from .hydra import parse_hydra_config
from ..data.datamodule import XRayDataModule
from ..model import instantiate_model
from ..model.experiment import Experiment
from ..utils import silence_pydicom_warnings


def run() -> None:
    parser = setup_common_cli_parser()
    args = parser.parse_args()
    config = parse_hydra_config(args)

    silence_pydicom_warnings()

    checkpoints = ModelCheckpoint(monitor='mAP', filename='{epoch}_{mAP:.3f}', save_top_k=5, mode='max')
    trainer = Trainer.from_argparse_args(args, callbacks=[checkpoints, LearningRateMonitor()])
    data = XRayDataModule(config.data)
    model = instantiate_model(config.model.num_classes, config.model.trainable_backbone_layers)
    experiment = Experiment(model, config)
    trainer.fit(experiment, datamodule=data)
    report_checkpoints(checkpoints)
    predictions = trainer.predict(experiment, datamodule=data)
    parsed_predictions = parse_predictions(predictions, data.predict_dataset.image_ids)
    generate_submission_file(config.submission.file, parsed_predictions, config.evaluation.confidence_threshold)


@rank_zero_only
def report_checkpoints(checkpointing_callback: ModelCheckpoint) -> None:
    print(f'Best mAP was {checkpointing_callback.best_model_score}')
    print(f'Path to the best checkpoint is {checkpointing_callback.best_model_path}')


if __name__ == '__main__':
    run()
