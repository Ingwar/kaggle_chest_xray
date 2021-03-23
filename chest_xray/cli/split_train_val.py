from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_metadata(original_metadata: pd.DataFrame, train_fraction: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    image_ids = pd.unique(original_metadata['image_id'])
    train_image_ids, val_image_ids = train_test_split(image_ids, train_size=train_fraction, random_state=random_state)
    val_image_ids = pd.Series(val_image_ids, name='image_id')
    train_image_ids = pd.Series(train_image_ids, name='image_id')
    train_metadata = original_metadata.merge(train_image_ids, on='image_id')
    val_metadata = original_metadata.merge(val_image_ids, on='image_id')
    return train_metadata, val_metadata


def main() -> None:
    parser = ArgumentParser(description='Split annotated metadata to train and val sets')
    parser.add_argument('original_file', type=Path)
    parser.add_argument('new_train_file', type=Path)
    parser.add_argument('new_val_file', type=Path)
    parser.add_argument('--train-fraction', type=float, default=0.9)
    parser.add_argument('--random-state', type=int, default=13)

    args = parser.parse_args()
    original_metadata = pd.read_csv(args.original_file)
    train_metadata, validation_metadata = split_metadata(original_metadata, args.train_fraction, args.random_state)
    train_metadata.to_csv(args.new_train_file, index=False)
    validation_metadata.to_csv(args.new_val_file, index=False)


if __name__ == '__main__':
    main()
