import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import pandas as pd
import numpy as np
#from data import OneDim_to_ThreeDim_Converter
from grid2graph.useful_custom_functions import OneDim_to_ThreeDim_Converter

class GridDataset(Dataset):
    """Geometric Grids dataset."""

    def __init__(self, csv_file, csv_file_labels, clear_labels=False, transform=None, pre_transform=None,
                 name='NoNameGridDataset'):
        '''
        Args:
            csv_file (string): Path to the .csv file which contains pd.DataFrames in 1D.
            csv_file_labels (string): Path to the .csv file which contains labels of grids.
            transform (callable, optional): Optional transform to be applied on a sample.
            
        '''
        self.data_set = pd.read_csv(csv_file)
        self.label = pd.read_csv(csv_file_labels)
        if clear_labels:
            self.label = self.label.iloc[np.where(self.label.values[:, 0] != -1)[0]]
        self.transform = transform
        self.pre_transform = pre_transform
        self.name = name

    def __len__(self):
        return len(self.data_set.columns)  # returns the size of the dataset (nomber of columnss)

    def __getitem__(self, idx):
        grid = self.data_set.iloc[:, idx].values
        grid = grid.reshape(1, -1)
        grid_name = [self.data_set.columns[idx]]
        label = self.label.iloc[:, idx].values

        if self.transform:
            grid = self.transform(grid)
            label = torch.from_numpy(label)
            # add labels to every node
            tmp_tensor = torch.zeros(label.shape[0], dtype=torch.float64)
            label = torch.add(tmp_tensor, label)

        # x, edge, label = self.remove_none(grid['x'], grid['edge_index'], label)
        sample = Data(x=grid['x'], edge_index=grid['edge_index'], label=label)  # no grid_name
        return sample

    def stats(self):
        '''mean and std of entire dataset'''
        mean = self.data_set.values.mean()
        std = self.data_set.values.std()
        stats = {'mean': mean, 'std': std}
        return stats

    def _download(self):
        pass
    # def _process(self):
    # pass


class MultyProps_GridDataset(Dataset):
    """Geometric Grids dataset."""

    def __init__(self, tensor_file, csv_file_labels, clear_labels=False, transform=None, pre_transform=None,
                 clear=False, name='NoNameGridDataset'):
        '''
        Args:
            tensor_file (string): Path to the .pt file which contains multidimensional tensors.
            csv_file_labels (string): Path to the .csv file which contains labels of grids.
            transform (callable, optional): Optional transform to be applied on a sample.
            
        '''
        self.data_set = torch.load(tensor_file)
        self.label = pd.read_csv(csv_file_labels)
        if clear_labels:
            self.label = self.label.iloc[np.where(self.label.values[:, 0] != -1)[0]]
        self.transform = transform
        self.pre_transform = pre_transform
        self.name = name

    def __len__(self):
        return self.data_set.shape[0]  # returns the size of the dataset (nomber of columns)

    def __getitem__(self, idx):
        grid = self.data_set[idx]
        grid = grid.t().contiguous()
        # print('hi', grid)
        # grid_name =[self.data_set.columns[idx]]
        label = self.label.iloc[:, idx].values

        if self.transform:
            grid = self.transform(grid)
            label = torch.from_numpy(label)
            # add labels to every node
            tmp_tensor = torch.zeros(label.shape[0], dtype=torch.float64)
            label = torch.add(tmp_tensor, label)

        # sample = Data(x=grid['x'], edge_index=grid['edge_index'], label=label, grid_name=grid_name) # with grid_name
        # x, edge, label = self.remove_none(grid['x'], grid['edge_index'], label)
        sample = Data(x=grid['x'], edge_index=grid['edge_index'], label=label)  # no grid_name
        return sample

    def stats(self):
        '''mean and std of entire dataset'''
        mean = self.data_set.values.mean()
        std = self.data_set.values.std()
        stats = {'mean': mean, 'std': std}
        return stats

    def _download(self):
        pass
    # def _process(self):
    # pass


class MultyProps_GridDataset_zcorn(Dataset):
    """Geometric Grids dataset."""

    def __init__(self, tensor_file, csv_file_labels, zcorn_file, clear_labels=False, transform=None, pre_transform=None,
                 clear=False, name='NoNameGridDataset'):
        '''
        Args:
            tensor_file (string): Path to the .pt file which contains multidimensional tensors.
            csv_file_labels (string): Path to the .csv file which contains labels of grids.
            transform (callable, optional): Optional transform to be applied on a sample.

        '''
        self.data_set = torch.load(tensor_file)
        self.label = pd.read_csv(csv_file_labels)
        self.zcorn = np.load(zcorn_file)
        if clear_labels:
            self.label = self.label.iloc[np.where(self.label.values[:, 0] != -1)[0]]
        self.transform = transform
        self.pre_transform = pre_transform
        self.name = name

    def __len__(self):
        return self.data_set.shape[0]  # returns the size of the dataset (nomber of columns)

    def __getitem__(self, idx):
        grid = self.data_set[idx]
        grid = grid.t().contiguous()
        zcorn = self.zcorn[idx]
        # print('hi', grid)
        # grid_name =[self.data_set.columns[idx]]
        label = self.label.iloc[:, idx].values

        if self.transform:
            grid = self.transform((grid, zcorn))
            label = torch.from_numpy(label)
            # add labels to every node
            tmp_tensor = torch.zeros(label.shape[0], dtype=torch.float64)
            label = torch.add(tmp_tensor, label)

        # sample = Data(x=grid['x'], edge_index=grid['edge_index'], label=label, grid_name=grid_name) # with grid_name
        # x, edge, label = self.remove_none(grid['x'], grid['edge_index'], label)
        sample = Data(x=grid['x'], edge_index=grid['edge_index'], label=label)  # no grid_name
        return sample

    def stats(self):
        '''mean and std of entire dataset'''
        mean = self.data_set.values.mean()
        std = self.data_set.values.std()
        stats = {'mean': mean, 'std': std}
        return stats

    def _download(self):
        pass
    # def _process(self):
    # pass


# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class RescaleTo3D(object):
    """Rescale the grid to a given size in 3D array.

    Args:
        i_max - dimension in X
        j_max - dimension in Y
        k_max - dimension in Z

    Return:
        ThreeDim - 3d np.array  
        
    """

    def __init__(self, i_max, j_max, k_max):
        self.i_max = i_max
        self.j_max = j_max
        self.k_max = k_max

    def __call__(self, sample):
        # приводим массив к 3D размерности грида.
        Three_Dim = np.array(np.zeros((self.i_max, self.j_max, self.k_max)))
        i = 0
        j = 0
        k = 0
        z = 0
        while k <= self.k_max - 1:
            while j <= self.j_max - 1:
                while i <= self.i_max - 1:
                    Three_Dim[i, j, k] = sample[:, z]
                    i += 1
                    z += 1
                i = 0
                j += 1
            j = 0
            k += 1
        return (Three_Dim)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return (torch.from_numpy(sample))


class Normalize(object):
    '''Normalize tensor to [-1,1]. made it by myself, cause unable to use transforms.Normalize. Don't know why'''

    def __call__(self, sample):
        sample = sample.sub_(train_dataset.stats()['mean']).div_(train_dataset.stats()['std'])
        return (sample)


class GridToGraph_6(object):
    """Converts grid to Graph with 6 nearest neighbours.
       1) x > 
       2) < x
       3) ^ x
       4) v x
       5) x up
       6) x down
       
    Args:
        i_max - dimension in X
        j_max - dimension in Y
        k_max - dimension in Z

    Return:
        Graph - edges and features of nodes   
        
    """

    def __init__(self, i_max, j_max, k_max, clear=False):
        self.i_max = i_max
        self.j_max = j_max
        self.clear = clear
        self.k_max = k_max

    def remove_none(self, x, edges):
        label_idx = torch.arange(x.shape[0])
        for j in range(x.shape[0]):
            v = x[j]
            if (v == -1).any():
                label_idx[j:] -= 1
                label_idx[j] = -1

        edges[0] = label_idx[edges[0]]
        edges[1] = label_idx[edges[1]]
        idx = torch.where((edges[0] != -1) & (edges[1] != -1))
        edges = torch.stack([edges[0][idx], edges[1][idx]])
        return x[torch.where(label_idx != -1)], edges

    def get(self, i_max, j_max, k_max):
        limit = i_max * j_max * k_max
        edge_start, _ = torch.arange(limit).repeat(6).sort()
        edge_finish = (torch.ones(limit * 6) * -1).long()

        idx1 = torch.where(edge_start[::6] < (i_max * (edge_start[::6] // i_max + 1)) - 1)
        edge_finish[::6][idx1] = edge_start[::6][idx1] + 1  # right

        idx2 = torch.where(edge_start[1::6] > (i_max * (edge_start[1::6] // i_max)))
        edge_finish[1::6][idx2] = edge_start[1::6][idx2] - 1  # left

        idx3 = torch.where(edge_start[2::6] >= (i_max * ((j_max * (edge_start[2::6] // (i_max * j_max))) + 1)))
        edge_finish[2::6][idx3] = edge_start[2::6][idx3] - i_max  # backward

        idx4 = torch.where(edge_start[3::6] < (i_max * ((j_max * (edge_start[3::6] // (i_max * j_max) + 1)) - 1)))
        edge_finish[3::6][idx4] = edge_start[3::6][idx4] + i_max  # forward

        idx5 = torch.where(edge_start[4::6] >= (i_max * j_max))
        edge_finish[4::6][idx5] = edge_start[4::6][idx5] - (i_max * j_max)  # above

        idx6 = torch.where(edge_start[5::6] < (i_max * j_max * (k_max - 1)))
        edge_finish[5::6][idx6] = edge_start[5::6][idx6] + (i_max * j_max)  # below

        edge_start = edge_start[torch.where(edge_finish != -1)]
        edge_finish = edge_finish[torch.where(edge_finish != -1)]

        return torch.stack((edge_start, edge_finish))

    def __call__(self, sample):

        edge_index = self.get(self.i_max, self.j_max, self.k_max)

        features_of_nodes = torch.transpose(torch.Tensor(sample), 0, 1)
        true_x = features_of_nodes
        true_edge = edge_index
        if self.clear:
            true_x, true_edge = self.remove_none(features_of_nodes, edge_index)
        Graph = Data(x=true_x, edge_index=true_edge, www=['place_for_string_data'])
        return Graph


class GridToGraph_zcorn(object):
    """Converts grid to Graph with 4 + multiple left-right neighbours, depending on the z coordinates.
       1) x >
       2) < x
       3) ^ x
       4) v x
       5) x up
       6) x down

    Args:
        i_max - dimension in X
        j_max - dimension in Y
        k_max - dimension in Z
        zcorn - array of z coordinates

    Return:
        Graph - edges and features of nodes

    """

    def __init__(self, i_max, j_max, k_max, clear=False):
        self.i_max = i_max
        self.j_max = j_max
        self.clear = clear
        self.k_max = k_max

    def remove_none(self, x, edges):
        label_idx = torch.arange(x.shape[0])
        for j in range(x.shape[0]):
            v = x[j]
            if (v == -1).any():
                label_idx[j:] -= 1
                label_idx[j] = -1

        edges[0] = label_idx[edges[0]]
        edges[1] = label_idx[edges[1]]
        idx = torch.where((edges[0] != -1) & (edges[1] != -1))
        edges = torch.stack([edges[0][idx], edges[1][idx]])
        return x[torch.where(label_idx != -1)], edges

    def get(self):
        limit = self.i_max * self.j_max * self.k_max
        edge_start, _ = torch.arange(limit).repeat(4).sort()
        edge_finish = (torch.ones(limit * 4) * -1).long()

        idx1 = torch.where(edge_start[::4] >= (self.i_max * ((self.j_max * (edge_start[::4] // (self.i_max * self.j_max))) + 1)))
        edge_finish[::4][idx1] = edge_start[::4][idx1] - self.i_max  # backward

        idx2 = torch.where(edge_start[1::4] < (self.i_max * ((self.j_max * (edge_start[1::4] // (self.i_max * self.j_max) + 1)) - 1)))
        edge_finish[1::4][idx2] = edge_start[1::4][idx2] + self.i_max  # forward

        idx3 = torch.where(edge_start[2::4] >= (self.i_max * self.j_max))
        edge_finish[2::4][idx3] = edge_start[2::4][idx3] - (self.i_max * self.j_max)  # above

        idx4 = torch.where(edge_start[3::4] < (self.i_max * self.j_max * (self.k_max - 1)))
        edge_finish[3::4][idx4] = edge_start[3::4][idx4] + (self.i_max * self.j_max)  # below

        edge_start = edge_start[torch.where(edge_finish != -1)]
        edge_finish = edge_finish[torch.where(edge_finish != -1)]

        return torch.stack((edge_start, edge_finish))

    def get_zcorn(self, zcorn):

        cubes = torch.zeros(self.i_max, self.j_max, self.k_max, 8)
        k = 0
        for plane in range(self.k_max):
            for column in range(self.j_max):
                for line in range(self.i_max):
                    cubes[line, column, plane, 0] = zcorn[k]
                    k += 1
                    cubes[line, column, plane, 1] = zcorn[k]
                    k += 1
                for line in range(self.i_max):
                    cubes[line, column, plane, 2] = zcorn[k]
                    k += 1
                    cubes[line, column, plane, 3] = zcorn[k]
                    k += 1

            for column in range(self.j_max):
                for line in range(self.i_max):
                    cubes[line, column, plane, 4] = zcorn[k]
                    k += 1
                    cubes[line, column, plane, 5] = zcorn[k]
                    k += 1
                for line in range(self.i_max):
                    cubes[line, column, plane, 6] = zcorn[k]
                    k += 1
                    cubes[line, column, plane, 7] = zcorn[k]
                    k += 1

        def add_left(cube, left_column_idx, left_column_coord):
            res = []
            for i in range(len(left_column_idx)):
                other_cube = left_column_coord[i]
                if (cube[4] > other_cube[5] > cube[0]) or (cube[6] > other_cube[7] > cube[2]) \
                        or (cube[4] > other_cube[1] > cube[0]) or (cube[6] > other_cube[3] > cube[2]) \
                        or (other_cube[1] == cube[0] and other_cube[5] == cube[4]):
                    res.append(left_column_idx[i])
            return res

        def add_right(cube, right_column_idx, right_column_coord):
            res = []
            for i in range(len(right_column_idx)):
                other_cube = right_column_coord[i]
                if (cube[5] > other_cube[4] > cube[1]) or (cube[7] > other_cube[6] > cube[3]) \
                        or (cube[5] > other_cube[0] > cube[1]) or (cube[7] > other_cube[2] > cube[3]) \
                        or (other_cube[0] == cube[1] and other_cube[4] == cube[5]):
                    res.append(right_column_idx[i])
            return res

        edge_index_first = []
        edge_index_second = []
        grid_idx = OneDim_to_ThreeDim_Converter(self.i_max, self.j_max, self.k_max, torch.arange(self.i_max * self.j_max * self.k_max)).astype(int)
        for x in range(self.i_max):
            for y in range(self.j_max):
                for z in range(self.k_max):
                    left_nbrs = []
                    right_nbrs = []
                    if x < self.i_max - 1:
                        right_nbrs = add_right(cubes[x, y, z], grid_idx[x + 1, y, :], cubes[x + 1, y, :])
                    if x > 0:
                        left_nbrs = add_left(cubes[x, y, z], grid_idx[x - 1, y, :], cubes[x - 1, y, :])
                    edge_index_first.extend([grid_idx[x, y, z]] * (len(right_nbrs) + len(left_nbrs)))
                    edge_index_second.extend(right_nbrs)
                    edge_index_second.extend(left_nbrs)

        return torch.stack((torch.tensor(edge_index_first),
                            torch.tensor(edge_index_second)))

    def __call__(self, sample):
        zcorn = sample[1]
        sample = sample[0]
        first = self.get()
        second = self.get_zcorn(zcorn)

        edge_index = torch.cat((first, second), dim=1)

        features_of_nodes = torch.transpose(torch.Tensor(sample), 0, 1)
        true_x = features_of_nodes
        true_edge = edge_index
        if self.clear:
            true_x, true_edge = self.remove_none(features_of_nodes, edge_index)
        Graph = Data(x=true_x, edge_index=true_edge, www=['place_for_string_data'])
        return Graph
