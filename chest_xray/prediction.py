from csv import DictWriter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np

__all__ = [
    'generate_submission_file',
    'parse_predictions',
    'Prediction',
]

NO_FINDING_TARGET_STRING = '14 1.0 0 0 1 1'


@dataclass(frozen=True)
class Prediction:
    image_id: str
    box: np.ndarray
    label: int
    score: float


def parse_predictions(predictions: List[Tuple[np.ndarray, List[Dict[str, np.ndarray]]]], image_ids: List[str]) -> Iterable[Prediction]:
    for batch in predictions:
        indices, batch_predictions = batch
        for index, prediction in zip(indices, batch_predictions):
            boxes = prediction['boxes']
            labels = prediction['labels']
            scores = prediction['scores']
            image_id = image_ids[index]
            for box, label, score in zip(boxes, labels, scores):
                yield Prediction(image_id, box, label, score)


def generate_submission_file(path: Union[Path, str], predictions: Iterable[Prediction]) -> None:
    # TODO: Add threshold filtration
    id_column = 'image_id'
    prediction_column = 'PredictionString'
    with open(path, 'w') as csv_file:
        writer = DictWriter(csv_file, fieldnames=[id_column, prediction_column])
        writer.writeheader()
        for p in predictions:
            target_string = f'{p.label - 1} {p.score} {p.box[0]} {p.box[1]} {p.box[2]} {p.box[3]}'
            writer.writerow({id_column: p.image_id, prediction_column: target_string})
