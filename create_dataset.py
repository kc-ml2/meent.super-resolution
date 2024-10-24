import torch
torch.manual_seed(0)

import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import meent

torch.set_default_dtype(torch.float64)



def generate_data(fto, dataset_size, pattern_size):
    rcwa_options = dict(backend=2, thickness=[205, ], period=[300, 300],
                        fto=fto,
                        n_top=1, n_bot=1,
                        wavelength=900,
                        pol=0.5,
                        )
    pattern_list = torch.zeros((dataset_size, pattern_size))
    rayleigh_list = torch.zeros((dataset_size, 4, 1, 2*fto+1), dtype=torch.complex128)

    for i in range(dataset_size):
        if i % 100 == 0: print(i)

        # ucell = torch.tensor([[[1,0,1,1,1,1,0]]]) * 3 + 1
        ucell = torch.randint(0, 2, (1, 1, pattern_size)) * 3 + 1

        mee = meent.call_mee(**rcwa_options)
        mee.ucell = ucell

        result = mee.conv_solve()

        result_given_pol = result.res
        result_te_incidence = result.res_te_inc
        result_tm_incidence = result.res_tm_inc

        te_R = result_te_incidence.R_s
        te_T = result_te_incidence.T_s
        tm_R = result_tm_incidence.R_p
        tm_T = result_tm_incidence.T_p

        pattern_list[i] = ucell
        rayleigh_list[i, 0] = te_R
        rayleigh_list[i, 1] = te_T
        rayleigh_list[i, 2] = tm_R
        rayleigh_list[i, 3] = tm_T


    return pattern_list, rayleigh_list
    # class CreateDataset(Dataset):
    #     def __init__(self, X, Y):
    #         self.X = X
    #         self.Y = Y
    #
    #     def __len__(self):
    #         return self.X.shape[0]
    #
    #     def __getitem__(self, idx):
    #         return self.X[idx], self.Y[idx]
    #
    # ds = CreateDataset(pattern_list, rayleigh_list)
    # return ds


if __name__ == '__main__':

    pattern_size = 8*8
    dataset_size = int(1E6)
    ratio = 0.8
    fto_list = [3, 4, 5, 6]
    fto_list = [3]
    for i, fto in enumerate(fto_list):
        pattern_list, rayleigh_list = generate_data(fto=fto, dataset_size=dataset_size, pattern_size=pattern_size)
        torch.save(pattern_list[:int(dataset_size*ratio)], f'./pattern_list_fto_{fto}_train.pt')
        torch.save(rayleigh_list[:int(dataset_size*ratio)], f'./rayleigh_list_fto_{fto}_train.pt')

        torch.save(pattern_list[int(dataset_size*ratio):], f'./pattern_list_fto_{fto}_test.pt')
        torch.save(rayleigh_list[int(dataset_size*ratio):], f'./rayleigh_list_fto_{fto}_test.pt')

    print(0)
