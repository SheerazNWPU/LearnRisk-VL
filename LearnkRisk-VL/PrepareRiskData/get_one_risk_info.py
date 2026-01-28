from os.path import join

import numpy as np
import pandas as pd
from get_one_distance import get_one_distance
from get_one_knn_count import get_one_knn_count


def get_one_risk_info(
    data_dir, data_sets, cnns, layers, elem_name_str, elems, csv_dir_str, k_list
):
    # get risk elem
    for layer in layers:
        for cnn in cnns:
            print("=== geting one_risk_info of {}_{} ===".format(cnn, layer))

            elem_name = elem_name_str.format(cnn)
            csv_dir = csv_dir_str.format(data_dir, cnn)
            # Load targets
            targets_df = pd.read_csv(join(csv_dir, "targets_{}.csv".format(data_sets[0])), header=None)
            targets = targets_df.to_numpy().flatten()
            
            # Count unique classes
            num_class = len(np.unique(targets))
            print('The number of Class:  ')
            print(num_class)
            
            get_one_distance(layer, elem_name, num_class, csv_dir, "cosine", data_sets)
            get_one_knn_count(
                k_list, layer, elem_name, num_class, csv_dir, "cosine", data_sets
            )

    # Merge csv
    csv_path_list = []
    for cnn in cnns:
        for layer in layers:
            for elem in elems:
                
                if elem == 'fangcha':
                    if layer == 'x4':
                        continue
                    if cnn == 'CCT':
                        continue
                if elem == 'xs8' or elem == 'xs1' or elem == 'xs3' or elem == 'xs5':
                    if layer == 'x4':
                        continue
                    if cnn == 'CCT':
                        continue
                if elem == 'paddingdis':
                    if cnn == 'CCT':
                        continue
                if elem == 'padknn8' or elem == 'padknn1':
                    if cnn == 'CCT':
                        continue
                if elem=='xsdis':
                    if cnn == 'CCT':
                        continue
                    if layer == 'x4':
                        continue
                if elem == 'all3' or elem == 'all5':
                    if cnn == 'CCT':
                        continue
                    if layer == 'x4':
                        continue
                csv_path_list.append(
                    join(
                        csv_dir_str.format(data_dir, cnn),
                        "{}_{}_one_{}.csv".format(cnn, layer, elem),
                    )
                )
    
    all_info = pd.read_csv(csv_path_list[0], header=None).to_numpy()[:, :2]
    #print(csv_path_list)
    for csv_path in csv_path_list:
        #print(csv_path)
        csv = pd.read_csv(csv_path, header=None).to_numpy()[:, -1:]
        all_info = np.hstack((all_info, csv))

    pd.DataFrame(all_info).to_csv(
        "/home/15t/Gul/SG/sheeraz/result_archive/risk_elem/{}/DBLP-Scholar/pair_info_more.csv".format(data_dir), #change path to the save distribution
        header=None,
        index=None,
    )

    all_info = all_info.tolist()
    # for Amazon
    for data_set in ["train"]:
        temp_csv = [all_info[0]]
        
        for line in all_info[1:]:
            filename = line[0]
            #print(filename)
            
            # SIMPLE FIX: Just check if data_set is in the filename
            if data_set in filename:
                temp_csv.append(line)
        
        pd.DataFrame(temp_csv).to_csv(
            "/home/15t/Gul/SG/sheeraz/result_archive/risk_elem/{}/DBLP-Scholar/325/{}.csv".format(
                data_dir, data_set
            ),
            header=None,
            index=None,
        )
    # for data_set in ['train', 'val', 'test']:
    '''
    for data_set in ["train"]:
        temp_csv = [all_info[0]]
    
        for line in all_info[1:]:
            filename = line[0]
            print(filename)
            # For CIFAR-100: paths are like "train_000001.png", "val_000001.png", "test_000001.png"
            # Extract prefix before underscore
            if '_' in filename:
                filename_prefix = filename.split("_")[0]
                if filename_prefix == data_set:
                    temp_csv.append(line)
            else:
                # Fallback for paths with "/" separators (medical dataset format)
                try:
                    if line[0].split("/")[1] == data_set:
                        temp_csv.append(line)
                except IndexError:
                    # Skip if path doesn't have enough parts
                    continue
    
        pd.DataFrame(temp_csv).to_csv(
            "/home/15t/Gul/SG/sheeraz/result_archive/risk_elem/{}/DBLP-Scholar/325/{}.csv".format(
                data_dir, data_set
            ),
            header=None,
            index=None,
        )

      '''
if __name__ == "__main__":

    get_one_risk_info()
