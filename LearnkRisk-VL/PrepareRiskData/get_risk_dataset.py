import os
import shutil
from os.path import join

import numpy as np
import pandas as pd
from get_one_risk_info import get_one_risk_info
from get_risk_info import get_risk_info


def my_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# settings
# ['cars_100_5val', 'cars_75_5val',
# 'cub_100_5val', 'cub_75_5val',
# 'fgvc_100_5val', 'fgvc_75_5val',
# 'cars_100_10val', 'cars_75_10val', 'cars_50_10val', 'cars_25_10val',
# 'cub_100_10val', 'cub_75_10val', 'cub_50_10val', 'cub_25_10val',
# 'fgvc_100_10val', 'fgvc_75_10val', 'fgvc_50_10val', 'fgvc_25_10val']
data_dirs = ['Office31_']  # dataset dir
data_sets = ['train', 'val', "test"]
cnns = ['Amazon_deepseek'] #'r101', 'r50','d169','CCT'    # multiple CNNs used to get features
layers =['x4', 'xc', 'fused'] # ['x4', 'xc', 'mi', 'f', 'maxif', 'ka', 'pet', 'kt', 'sp']
elem_name_str = "{}"
elems = ["distance", 'knn5', 'knn7'] # ["distance", 'knn1', 'knn8','xs5','xs3']
csv_dir_str = "result_archive/{}{}"  # where you save the distribution csv

k_list = [5,7] # [1, 8]
archive_dir = "result_archive/risk_elem/"  # path to save risk feature csv
note = ''  # additional note to add in save folder, e.g., save 'fgvc_100_test' at archive_dir


# get risk elem
if __name__ == "__main__":

    for data_dir in data_dirs:
        print('\n===== {} ====='.format(data_dir))
        my_mkdir(join(archive_dir, data_dir))
        my_mkdir(join(archive_dir, data_dir, "risk_dataset{}".format(note)))
        my_mkdir(join(archive_dir, data_dir, "softmax"))
        my_mkdir(join(archive_dir, data_dir, "DBLP-Scholar"))
        my_mkdir(join(archive_dir, data_dir, "DBLP-Scholar", '325'))

        get_risk_info(
            data_dir, data_sets, cnns, layers, elem_name_str, elems, csv_dir_str, k_list, note
        )
        get_one_risk_info(
            data_dir, data_sets, cnns, layers, elem_name_str, elems, csv_dir_str, k_list
        )

        # copy softmax to final save dir
        for cnn in cnns:
            my_mkdir(join(archive_dir, data_dir, "softmax", "{}".format(cnn)))
            for data_set in data_sets:
                shutil.copy(
                    join(
                        csv_dir_str.format(data_dir, cnn),
                        "distribution_xc_{}.csv".format(data_set),
                    ),
                    join(
                        archive_dir,
                        data_dir,
                        "softmax",
                        "{}".format(cnn),
                        "distribution_xc_{}.csv".format(data_set),
                    ),
                )
