from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

from .args import setup_common_cli_parser
from .hydra import parse_hydra_config
from ..data.datamodule import XRayDataModule, XRayTrainOnlyDataModule
from ..model.experiment import Experiment
from ..utils import silence_pydicom_warnings


def run() -> None:
    parser = setup_common_cli_parser()
    args = parser.parse_args()
    config = parse_hydra_config(args)

    silence_pydicom_warnings()

    if config.data.validation is not None:
        data = XRayDataModule(config.data)
        checkpoints = ModelCheckpoint(
            monitor='mAP',
            filename='{epoch}_{mAP:.3f}',
            save_top_k=5,
            mode='max',
            save_last=True
        )
    else:
        data = XRayTrainOnlyDataModule(config.data)
        checkpoints = ModelCheckpoint(save_top_k=-1, save_last=True)
    trainer = Trainer.from_argparse_args(args, callbacks=[checkpoints, LearningRateMonitor()])
    experiment = Experiment(config)
    trainer.fit(experiment, datamodule=data)
    report_checkpoints(checkpoints)


@rank_zero_only
def report_checkpoints(checkpointing_callback: ModelCheckpoint) -> None:
    print(f'Best mAP was {checkpointing_callback.best_model_score}')
    print(f'Path to the best checkpoint is {checkpointing_callback.best_model_path}')


if __name__ == '__main__':
    run()
