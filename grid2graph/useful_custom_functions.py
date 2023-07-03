import torch
from sklearn.preprocessing import PowerTransformer
import numpy as np  # linear algebra


def OneDim_to_ThreeDim_Converter(i_max, j_max, k_max, One_Dim):
    # приводим массив к 3D размерности грида.
    Three_Dim = np.array(np.zeros((i_max, j_max, k_max)))
    i = 0
    j = 0
    k = 0
    z = 0

    while k <= k_max - 1:
        while j <= j_max - 1:
            while i <= i_max - 1:
                Three_Dim[i, j, k] = One_Dim[z]
                i += 1
                z += 1
            i = 0
            j += 1
        j = 0
        k += 1

    return Three_Dim


def make_me_properties_tensor(*args, file_name='tensor.pt'):
    """
    Args:
        * args - graph datasets (class GridDataset) of properties which shoul be gathered into one graph of various properties

    Return:
           file "Tensor.pt" for class MultyProps_GridDataset.
    """
    number_of_properties = len(args)
    number_of_models = len(args[0])
    number_of_nodes = args[0][0].num_nodes

    dataset_props_x = torch.zeros([number_of_models, number_of_nodes, number_of_properties],
                                  dtype=torch.float)  # create empty tensor for properties

    i = 0
    for i in range(len(args[0])):
        tuple_ = ((),)
        for j in range(len(args)):
            tuple_ = tuple_ + ((args[j][i]['x']),)
        j = 0
        print(i, end='\r')
        dataset_props_x[i] = torch.cat((tuple_[1:]), 1)
        # print(dataset_props_x[i])

    torch.save(dataset_props_x, file_name)
    print('you can find the tensor of properties in this file: ', file_name)