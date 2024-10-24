import torch

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

    class CreateDataset(Dataset):
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]

    ds = CreateDataset(pattern_list, rayleigh_list)
    return ds


def run(fto, dataset_size, pattern_size):
    dataset_train = generate_data(fto, dataset_size, pattern_size)
    dataset_test = generate_data(fto, 100, pattern_size)

    train_dataloader = DataLoader(dataset_train, batch_size=64)
    test_dataloader = DataLoader(dataset_test, batch_size=64)

    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self, fto_in_mlp, pattern_size):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(pattern_size, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 2*fto_in_mlp + 1)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork(fto, pattern_size)
    model.double()
    print(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, Y) in enumerate(dataloader):
            y = Y[:,0].squeeze().abs()
            # X = X.reshape((batch_size, -1))
            # y = Y[0].reshape((batch_size, -1)).abs()

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        # batch_size = 1
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, Y in dataloader:
                y = Y[:,0].squeeze().abs()
                # X = X.reshape((batch_size, -1))
                # y = Y[0].reshape((batch_size, -1)).abs()

            # for X, y in dataset:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
        test_loss /= num_batches
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    epochs = 1000
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")


if __name__ == '__main__':

    run(3, 10000, (8*8))

    print(0)
