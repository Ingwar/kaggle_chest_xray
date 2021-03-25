from argparse import ArgumentParser
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

from chest_xray.data.dataset import read_xray


def run() -> None:
    parser = ArgumentParser()
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    input_files = list(args.input_dir.glob('*.dicom'))
    args.output_dir.mkdir(exist_ok=True, parents=True)
    for image_file in tqdm(input_files):
        image_data = read_xray(image_file)
        image = Image.fromarray(image_data)
        image_name = image_file.stem
        image.save(args.output_dir / f'{image_name}.png')


if __name__ == '__main__':
    run()
