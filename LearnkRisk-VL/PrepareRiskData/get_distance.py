import math
import os
from os.path import join

import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as pairwise
from tqdm import tqdm, trange


def shorten_paths(paths):

    for i in range(len(paths)):
        paths[i] = "/".join(paths[i].split("/")[-4:])


# Calculate the center of high dimension data
def get_center(data_list):
    center = []

    for d in range(len(data_list[0])):  # d for dimension
        coordinate = 0

        for data in data_list:
            coordinate += data[d]

        coordinate /= len(data_list)
        center.append(coordinate)

    return center


def take_0(elem):
    return elem[0]


def point_min(list):
    min_index = list.index(min(list))

    for i in range(len(list)):

        if i == min_index:
            list[i] = 1
        else:
            list[i] = 0

    return list


def eval_distance(data_set, layer, labels, predictions, distances):
    # Evaluate prediction
    correct = 0
    for i in range(len(labels)):
        if predictions[i] == labels[i]:
            correct += 1

    acc = correct / len(labels)

    # Evaluate distance
    d_predictions = []

    for info in distances:
        # info = list(map(float, info))
        d_predictions.append(info.index(min(info)))

    d_correct = 0

    for i in range(len(labels)):
        if d_predictions[i] == labels[i]:
            d_correct += 1

    d_acc = d_correct / len(labels)

    # Calculate different wrong
    evaluation = []

    for i in range(len(labels)):
        evaluation.append([labels[i], predictions[i], d_predictions[i]])

    p_wrong = d_wrong = 0

    for data_labels in evaluation:

        if data_labels[1] != data_labels[2]:

            if data_labels[1] == data_labels[0]:
                d_wrong += 1
            elif data_labels[2] == data_labels[0]:
                p_wrong += 1

    # print('Dataset: {}, Layer: {}'.format(data_set, layer))
    # print('Acc, D_Acc, d_right_p_wrong, p_right_d_wrong')
    print("{:.2f}, {:.2f}, {}, {}".format(acc * 100, d_acc * 100, p_wrong, d_wrong))
    # print('{:.2f}'.format(d_acc * 100))


# Calculate the distance to each class center of every data
def get_distance(
    layer, elem_name, num_class, csv_dir, metric, data_sets=["train", "val", "test"]
):
    print("\n===== layer: {} =====".format(layer))
    print("Acc, D_Acc, d_right_p_wrong, d_wrong_p_right")
    # Calculate the class centers by train data
    print(os.path.join(csv_dir, "distribution_{}_{}.csv".format(layer, data_sets[0])))
    coordinate_train = pd.read_csv(
        os.path.join(csv_dir, "distribution_{}_{}.csv".format(layer, data_sets[0])),
        header=None,
    ).to_numpy()
    label = pd.read_csv(
        os.path.join(csv_dir, "targets_{}.csv".format(data_sets[0])), header=None
    ).to_numpy()
    # coordinate_train = pd.read_csv(os.path.join(csv_dir, 'distribution_{}_train_more.csv'.format(layer)), header=None).to_numpy()
    # label = pd.read_csv(os.path.join(csv_dir, 'labels_train_more.csv'), header=None).to_numpy()

    train_cls = []

    for cls in range(num_class):
        coordinate_cls = []

        for i in range(len(label)):

            if label[i][0] == cls:
                coordinate_cls.append(coordinate_train[i])

        train_cls.append(coordinate_cls)

    centers = []

    for i in range(len(train_cls)):
        centers.append(get_center(train_cls[i]))

    for data_set in data_sets:
        # for data_set in ['train_more', 'val_less', 'test']:

        # Read coordinate and label of data
        coordinates = pd.read_csv(
            os.path.join(csv_dir, "distribution_{}_{}.csv".format(layer, data_set)),
            header=None,
        ).to_numpy()

        labels = (
            pd.read_csv(
                os.path.join(csv_dir, "targets_{}.csv".format(data_set)), header=None
            )
            .to_numpy()
            .flatten()
        )
        paths = (
            pd.read_csv(
                os.path.join(csv_dir, "paths_{}.csv".format(data_set)), header=None
            )
            .to_numpy()
            .flatten()
        )
        shorten_paths(paths)
        predictions = (
            pd.read_csv(
                os.path.join(csv_dir, "predictions_{}.csv".format(data_set)),
                header=None,
            )
            .to_numpy()
            .flatten()
        )

        # Calculate the distance to each class center of every data
        # metric = euclidean, cosine, braycurtis, seuclidean, canberra
        distance_to_center = pairwise.pairwise_distances(
            coordinates, centers, metric=metric
        ).tolist()

        # pd.DataFrame(distance_to_center).to_csv(os.path.join(csv_dir, '{}_distance_{}.csv' \
        #                                                       .format(layer, data_set)), header=None, index=None)

        # Evaluate distance
        eval_distance(data_set, layer, labels, predictions, distance_to_center)

        # Insert path and label of data to csv
        print('dis')
        print(len(distance_to_center))
        print('path')
        print(len(paths))
        #distance_to_center.sort(key=take_0)
        for i in range(len(distance_to_center)):
            distance_to_center[i].insert(0, paths[i])
            distance_to_center[i].insert(1, labels[i])
        #distance_to_center.sort(key=take_0)
        #distance_to_center = np.array(distance_to_center, dtype=np.float32)
        
        #print(distance_to_center)

        exec("distance_to_center_{} = distance_to_center".format(data_set))

    # Merge 3 csvs together
    distance_center_all = []

    for data_set in data_sets:
        # for data_set in ['train_more', 'val_less', 'test']:
        exec("distance_center_all.extend(distance_to_center_{})".format(data_set))

    # Create the header of csv
    header = ["data", "label"]

    for i in range(num_class):
        header.append("{}_{}_class_{:0>3d}_distance".format(elem_name, layer, i))

    # Save the final csv
    distance_center_all.insert(0, header)
    pd.DataFrame(distance_center_all).to_csv(
        os.path.join(csv_dir, "{}_{}_distance.csv".format(elem_name, layer)),
        header=None,
        index=None,
    )

    # print('{}_{}_distance_to_center_all.csv saved at {}\n'.format(elem_name, layer, csv_dir))

    # # Get nearest class csv
    # n_distance_center_all = []
    #
    # for i in range(len(distance_center_all) - 1):
    #     info = distance_center_all[i + 1]
    #     raw_distance = info[2:]
    #     n_info = [info[0], info[1]]
    #     n_info.extend(point_min(raw_distance))
    #     n_distance_center_all.append(n_info)
    #
    # n_header = ['data', 'label']
    #
    # for i in range(num_class):
    #     n_header.append('{}_{}_class_{:0>3d}_is_nearest'.format(elem_name, layer, i))
    #
    # n_distance_center_all.insert(0, n_header)
    #
    # pd.DataFrame(n_distance_center_all).to_csv(os.path.join(csv_dir, '{}_{}_min_distance.csv' \
    #                                                       .format(elem_name, layer)), header=None, index=None)


# settings
metrics = [
    "cityblock",
    "cosine",
    "euclidean",
    "l1",
    "l2",
    "manhattan",
    "braycurtis",
    "canberra",
    "correlation",
    "minkowski",
    "seuclidean",
    "sqeuclidean",
]
# metrics = ['cosine', 'correlation', 'braycurtis', 'canberra', 'seuclidean']
# layer = 'x'
data_sets = ["train", "test"]

cnns = ["101", "50", "Wide50", "Densenet121"]
layers = ['x4', "xc"]
elem_name_str = "{}"
csv_dir_str = '/home/ssd0/SG/sheeraz/result_archive/Bracs_{}/'


if __name__ == "__main__":

    for layer in layers:

        for cnn in cnns:
            print("===== CNN: {} =====".format(cnn))

            elem_name = elem_name_str.format(cnn)
            csv_dir = csv_dir_str.format(cnn)
            num_class = (
                int(
                    pd.read_csv(join(csv_dir, "predictions_train.csv"), header=None)
                    .to_numpy()
                    .flatten()[-1]
                )
                + 1
            )
            get_distance(layer, elem_name, num_class, csv_dir, "cosine", data_sets)

    print("Done.")
