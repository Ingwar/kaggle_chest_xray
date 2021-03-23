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
    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


def parse_predictions(predictions: List[Tuple[np.ndarray, List[Dict[str, np.ndarray]]]], image_ids: List[str]) -> Iterable[Prediction]:
    for batch in predictions:
        indices, batch_predictions = batch
        for index, prediction in zip(indices, batch_predictions):
            boxes = prediction['boxes']
            labels = prediction['labels']
            scores = prediction['scores']
            image_id = image_ids[index]
            yield Prediction(image_id, boxes, labels, scores)


def generate_submission_file(path: Union[Path, str], predictions: Iterable[Prediction]) -> None:
    # TODO: Add threshold filtration
    id_column = 'image_id'
    prediction_column = 'PredictionString'
    with open(path, 'w') as csv_file:
        writer = DictWriter(csv_file, fieldnames=[id_column, prediction_column])
        writer.writeheader()
        for p in predictions:
            target_strings = []
            for box, label, score in zip(p.boxes, p.labels, p.scores):
                target_strings.append(f'{label - 1} {score} {box[0]} {box[1]} {box[2]} {box[3]}')
            target_string = ' '.join(target_strings)
            writer.writerow({id_column: p.image_id, prediction_column: target_string})
