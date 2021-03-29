from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


def run() -> None:
    parser = ArgumentParser()
    parser.add_argument('input_file', type=Path)
    parser.add_argument('output_file', type=Path)
    parser.add_argument('--no-finding-class', type=int, default=14)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    train_data = pd.read_csv(args.input_file)
    image_ids = pd.unique(train_data['image_id'])
    new_image_info = []
    for image_id in tqdm(image_ids):
        image_info = train_data[train_data['image_id'] == image_id].reset_index()
        no_finding_image_info = image_info[image_info['class_id'] == args.no_finding_class]
        if len(no_finding_image_info) > 1:
            if len(image_info) != len(no_finding_image_info):
                raise ValueError(f'Image {image_id} has both finding and no findings')
            image_info = image_info.drop(columns='rad_id').drop_duplicates(['image_id'])
        image_info_averaged = averageCoordinates(image_info, args.threshold)
        new_image_info.append(image_info_averaged)
    train_with_averaged_coordinates = pd.concat(new_image_info)
    train_with_averaged_coordinates.to_csv(args.output_file, index=False)


if __name__ == '__main__':
    run()


# Code below is taken from https://etrain.xyz/en/posts/vinbigdata-chest-x-ray-abnormalities-detection
# Only modification is that I don't use width and height info

def bb_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def averageCoordinates(df, threshold):
    tmp_df = df.reset_index()
    duplicate = {}
    for index1, row1 in tmp_df.iterrows():
        if index1 < len(tmp_df) - 1:
            next_index = index1 + 1
            for index2, row2 in tmp_df.loc[next_index:,:].iterrows():
                if row1["class_id"] == row2["class_id"]:
                    boxA = [row1['x_min'], row1['y_min'], row1['x_max'], row1['y_max']]
                    boxB = [row2['x_min'], row2['y_min'], row2['x_max'], row2['y_max']]
                    iou = bb_iou(boxA, boxB)
                    if iou > threshold:
                        if row1["index"] not in duplicate:
                            duplicate[row1["index"]] = []
                        duplicate[row1["index"]].append(row2["index"])

    remove_keys = []
    for k in duplicate:
        for i in duplicate[k]:
            if i in duplicate:
                for id in duplicate[i]:
                    if id not in duplicate[k]:
                        duplicate[k].append(id)
                if i not in remove_keys:
                    remove_keys.append(i)
    for i in remove_keys:
        del duplicate[i]

    rows = []
    removed_index = []
    for k in duplicate:
        row = tmp_df[tmp_df['index'] == k].iloc[0]
        X_min = [row['x_min']]
        X_max = [row['x_max']]
        Y_min = [row['y_min']]
        Y_max = [row['y_max']]
        removed_index.append(k)
        for i in duplicate[k]:
            removed_index.append(i)
            row = tmp_df[tmp_df['index'] == i].iloc[0]
            X_min.append(row['x_min'])
            X_max.append(row['x_max'])
            Y_min.append(row['y_min'])
            Y_max.append(row['y_max'])
        X_min_avg = sum(X_min) / len(X_min)
        X_max_avg = sum(X_max) / len(X_max)
        Y_min_avg = sum(Y_min) / len(Y_min)
        Y_max_avg = sum(Y_max) / len(Y_max)
        new_row = [row['image_id'], row['class_name'], row['class_id'], X_min_avg, Y_min_avg, X_max_avg, Y_max_avg]
        rows.append(new_row)

    for index, row in tmp_df.iterrows():
        if row['index'] not in removed_index:
            new_row = [row['image_id'], row['class_name'], row['class_id'], row['x_min'], row['y_min'], row['x_max'], row['y_max']]
            rows.append(new_row)

    new_df = pd.DataFrame(rows, columns =['image_id', 'class_name', 'class_id', 'x_min', 'y_min', 'x_max', 'y_max'])
    return new_df
