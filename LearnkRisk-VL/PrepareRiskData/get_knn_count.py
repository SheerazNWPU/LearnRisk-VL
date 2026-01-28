import multiprocessing
import os
from functools import partial
from itertools import combinations
from os.path import join

import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as pairwise
from tqdm import tqdm, trange


def shorten_paths(paths):

    for i in range(len(paths)):
        paths[i] = "/".join(paths[i].split("/")[-4:])


def point_max(list):
    max_index = list.index(max(list))

    for i in range(len(list)):

        if i == max_index:
            list[i] = 1
        else:
            list[i] = 0

    return list


def get_knn(k, list, label_train):
    list_sorted = sorted(list)
    knn = []

    for i in range(k):
        index = list.index(list_sorted[i])
        knn.append(label_train[index])

    return knn

def get_knn1(k, list, label_train):
    list_sorted = sorted(list)
    knn = []

    for i in range(k):
        index = list.index(list_sorted[i+1])
        knn.append(label_train[index])

    return knn


def eval_knn(data_set, layer, labels, predictions, count):
    # Evaluate prediction
    correct = 0
    for i in range(len(labels)):
        if predictions[i] == labels[i]:
            correct += 1

    acc = correct / len(labels)

    # Evaluate distance
    k_predictions = []

    for info in count:
        # info = list(map(int, info))
        k_predictions.append(info.index(max(info)))

    k_correct = 0

    for i in range(len(labels)):
        if k_predictions[i] == labels[i]:
            k_correct += 1

    k_acc = k_correct / len(labels)

    # Calculate different wrong
    evaluation = []

    for i in range(len(labels)):
        evaluation.append([labels[i], predictions[i], k_predictions[i]])

    p_wrong = k_wrong = 0

    for data_labels in evaluation:

        if data_labels[1] != data_labels[2]:

            if data_labels[1] == data_labels[0]:
                k_wrong += 1
            elif data_labels[2] == data_labels[0]:
                p_wrong += 1

    # print('Dataset: {}, Layer: {}'.format(data_set, layer))
    # print('Acc, K_Acc, k_right_p_wrong, k_wrong_p_right')
    print("{:.2f}, {:.2f}, {}, {}".format(acc * 100, k_acc * 100, p_wrong, k_wrong))
    # print('{:.2f}'.format(k_acc * 100))


# Calculate the knn_count to each class center of every data
def get_knn_count(
    k_list,
    layer,
    elem_name,
    num_class,
    csv_dir,
    metric,
    data_sets=["train", "val", "test"],
):
    print("\n===== layer: {} =====".format(layer))

    for k in k_list:
        print("\n===== K: {} =====".format(k))
        print("Acc, K_Acc, k_right_p_wrong, k_wrong_p_right")
        count_all = []

        # Get knn_count for all data_sets
        for data_set in data_sets:
            # for data_set in ['train_more', 'val_less', 'test']:
            # labels_train = pd.read_csv(join(csv_dir, 'labels_train.csv'), header=None).to_numpy().flatten()
            # distribution_train = pd.read_csv(join(csv_dir, 'distribution_{}_train.csv'.format(layer)), header=None).to_numpy()
            labels_train = (
                pd.read_csv(
                    join(csv_dir, "targets_{}.csv".format(data_sets[0])), header=None
                )
                .to_numpy()
                .flatten()
            )
            distribution_train = pd.read_csv(
                join(csv_dir, "distribution_{}_{}.csv".format(layer, data_sets[0])),
                header=None,
            ).to_numpy()

            paths = (
                pd.read_csv(join(csv_dir, "paths_{}.csv".format(data_set)), header=None)
                .to_numpy()
                .flatten()
            )
            shorten_paths(paths)
            labels = (
                pd.read_csv(
                    join(csv_dir, "targets_{}.csv".format(data_set)), header=None
                )
                .to_numpy()
                .flatten()
            )
            predictions = (
                pd.read_csv(
                    join(csv_dir, "predictions_{}.csv".format(data_set)), header=None
                )
                .to_numpy()
                .flatten()
            )

            distribution = pd.read_csv(
                join(csv_dir, "distribution_{}_{}.csv".format(layer, data_set)),
                header=None,
            ).to_numpy()

            # Get pairwise_distances for all
            # metric = euclidean, cosine, braycurtis
            pairwise_distances = pairwise.pairwise_distances(
                distribution, distribution_train, metric=metric
            ).tolist()

            # Get knn for all
            knn = []

            for i in range(len(pairwise_distances)):
                if data_set=='train':
                  knn.append(get_knn1(k, pairwise_distances[i], labels_train))
                else:
                    knn.append(get_knn(k, pairwise_distances[i], labels_train))

            # Get knn count for all
            count = []

            for i in range(len(knn)):
                line = np.bincount(knn[i], minlength=num_class).tolist()
                count.append(line)

            # Evaluate knn_count
            # print('===== K: {} ====='.format(k))
            eval_knn(data_set, layer, labels, predictions, count)

            # Combine all knn_count
            for i in range(len(count)):
                count[i].insert(0, paths[i])
                count[i].insert(1, labels[i])

            count_all.extend(count)

        # Create the header of csv
        header = ["data", "label"]

        for i in range(num_class):
            header.append("{}_{}_class_{:0>3d}_count{}".format(elem_name, layer, i, k))

        # Save the final csv
        count_all.insert(0, header)
        pd.DataFrame(count_all).to_csv(
            os.path.join(csv_dir, "{}_{}_knn{}.csv".format(elem_name, layer, k)),
            header=None,
            index=None,
        )

        # # Get most knn class csv
        # m_count_all = []
        #
        # for i in range(len(count_all) - 1):
        #     info = count_all[i + 1]
        #     raw_count = info[2:]
        #     m_info = [info[0], info[1]]
        #     m_info.extend(point_max(raw_count))
        #     m_count_all.append(m_info)
        #
        # m_header = ['data', 'label']
        #
        # for i in range(num_class):
        #     m_header.append('{}_{}_class_{:0>3d}_is_most'.format(elem_name, layer, i))
        #
        # m_count_all.insert(0, m_header)
        #
        # pd.DataFrame(m_count_all).to_csv(os.path.join(csv_dir, '{}_{}_most_knn{}.csv' \
        #                                                         .format(elem_name, layer, k)), header=None, index=None)


# settings
metrics = [
    "cityblock",
    "cosine",
    "euclidean",
    "l1",
    "l2",
    "manhattan",
    "nan_euclidean",
    "braycurtis",
    "canberra",
    "correlation",
    "minkowski",
    "seuclidean",
    "sqeuclidean",
]
# metrics = ['braycurtis', 'canberra', 'seuclidean']
# layer = 'x'
data_sets = ["train", "test"]

cnns = ["r101"]
layers = ['x4', "xc"]
elem_name_str = "{}"
csv_dir_str = '/home/ssd0/SG/sheeraz/result_archive/Bracs_Resnet101/'


k_list = [1, 8]


if __name__ == "__main__":
    get_knn_count()

    # for layer in layers:
    #
    #     for cnn in cnns:
    #         print("===== CNN: {} =====".format(cnn))
    #
    #         elem_name = elem_name_str.format(cnn)
    #         csv_dir = csv_dir_str.format(cnn)
    #         num_class = (
    #             int(
    #                 pd.read_csv(join(csv_dir, "predictions_train.csv"), header=None)
    #                 .to_numpy()
    #                 .flatten()[-1]
    #             )
    #             + 1
    #         )
    #         get_knn_count(
    #             k_list, layer, elem_name, num_class, csv_dir, "cosine", data_sets
    #         )
