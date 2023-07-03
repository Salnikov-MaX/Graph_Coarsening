import grid2graph.classGridDataset as classGridDataset
from torchvision import transforms
import grid2graph.useful_custom_functions as useful_custom_functions
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from  grid2graph.useful_custom_functions import OneDim_to_ThreeDim_Converter


def get_dset(path, path_labels, path_prefix, i_max=16, j_max=12, z_max=10):
    dataset = classGridDataset.GridDataset(path_prefix + path,
                                           path_prefix + path_labels, clear_labels=True,
                                           transform=transforms.Compose(
                                               [classGridDataset.GridToGraph_6(i_max, j_max, z_max)]),
                                           name='SRM_GridDataset')
    return dataset


def get_multi_dataset(path, path_labels, clear, path_prefix, i_max=16, j_max=12, z_max=10):
    dataset = classGridDataset.MultyProps_GridDataset(path,
                                                      path_prefix + path_labels,
                                                      clear_labels=True,
                                                      transform=transforms.Compose(
                                                          [classGridDataset.GridToGraph_6(i_max, j_max, z_max,
                                                                                          clear=clear)]),
                                                      name='SRM_GridDataset')
    return dataset


def get_multi_dataset_zcorn(path, path_labels, clear, path_prefix, path_zcorn, i_max=16, j_max=13, z_max=10):
    dataset = classGridDataset.MultyProps_GridDataset_zcorn(path,
                                                            path_prefix + path_labels, path_zcorn,
                                                            clear_labels=True,
                                                            transform=transforms.Compose(
                                                                [classGridDataset.GridToGraph_zcorn(i_max, j_max, z_max,
                                                                                                    clear=clear)]),
                                                            name='SRM_GridDataset')
    return dataset


from joblib import Parallel, delayed


def get_dataset(name, tensor_path='../DATA/processed_dir/Channels_1500_tensor.pt',
                device=torch.device("cuda:0" if torch.cuda.is_available()
                                    else "cpu"), i_max=16, j_max=13, z_max=10):
    dataset_perm_train = get_dset("Permeability.csv", "Facies.csv", name, i_max=i_max, j_max=j_max, z_max=z_max)
    dataset_poro_train = get_dset("Porosity.csv", "Facies.csv", name, i_max=i_max, j_max=j_max, z_max=z_max)

    useful_custom_functions.make_me_properties_tensor(dataset_poro_train, dataset_perm_train, file_name=tensor_path)
    dataset_train = get_multi_dataset(tensor_path, "Facies.csv", True, name, i_max=i_max, j_max=j_max, z_max=z_max)
    dataset_list = []

    for d in tqdm(dataset_train):
        dataset_list.append(d.to(device))

    return dataset_list


def get_dataset_zcorn(name, zcorn_path='zcorn.npy', tensor_path='../DATA/processed_dir/Fault_dataset.pt',
                      device=torch.device("cuda:0" if torch.cuda.is_available()
                                          else "cpu"), i_max=16, j_max=13, z_max=10):
    dataset_perm_train = get_dset("Permeability.csv", "Facies.csv", name, i_max=i_max, j_max=j_max, z_max=z_max)
    dataset_poro_train = get_dset("Porosity.csv", "Facies.csv", name, i_max=i_max, j_max=j_max, z_max=z_max)
    useful_custom_functions.make_me_properties_tensor(dataset_poro_train, dataset_perm_train, file_name=tensor_path)
    dataset_train = get_multi_dataset_zcorn(tensor_path, "Facies.csv", True, name, zcorn_path)
    dataset_list = []

    for d in tqdm(dataset_train):
        dataset_list.append(d.to(device))

    return dataset_list
