import torch
torch.set_default_dtype(torch.float64)

import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


writer = SummaryWriter()

import meent


def run(fto, dataset_size, pattern_size):

    class CreateDataset(Dataset):
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]

    pattern_list_fto_3_train = torch.load('./pattern_list_fto_3_train.pt')
    rayleigh_list_fto_3_train = torch.load('./rayleigh_list_fto_3_train.pt')
    pattern_list_fto_3_test = torch.load('./pattern_list_fto_3_test.pt')
    rayleigh_list_fto_3_test = torch.load('./rayleigh_list_fto_3_test.pt')

    dataset_3_train = CreateDataset(pattern_list_fto_3_train, rayleigh_list_fto_3_train)
    dataset_3_test = CreateDataset(pattern_list_fto_3_test, rayleigh_list_fto_3_test)

    pattern_list_fto_4_train = torch.load('./pattern_list_fto_4_train.pt')
    rayleigh_list_fto_4_train = torch.load('./rayleigh_list_fto_4_train.pt')
    pattern_list_fto_4_test = torch.load('./pattern_list_fto_4_test.pt')
    rayleigh_list_fto_4_test = torch.load('./rayleigh_list_fto_4_test.pt')

    dataset_4_train = CreateDataset(pattern_list_fto_4_train, rayleigh_list_fto_4_train)
    dataset_4_test = CreateDataset(pattern_list_fto_4_test, rayleigh_list_fto_4_test)

    train_dataloader = DataLoader(dataset_3_train, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(dataset_3_test, batch_size=64, shuffle=True)

    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self, fto_in_mlp, pattern_size):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(pattern_size, 512),
                nn.ReLU(),
                nn.Linear(512, 2**10),
                nn.ReLU(),
                nn.Linear(2**10, 2**10),
                nn.ReLU(),
                nn.Linear(2**10, 2*fto_in_mlp + 1)
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

            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 10 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        test_loss = 0

        with torch.no_grad():
            for X, Y in dataloader:
                y = Y[:,0].squeeze().abs()
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
        test_loss /= len(dataloader)
        writer.add_scalars("run1", {'loss-train':test_loss})
        # writer.add_scalar("run1", {'a':test_loss})

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0

        with torch.no_grad():
            for X, Y in dataloader:
                y = Y[:,0].squeeze().abs()
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
        test_loss /= num_batches
        writer.add_scalars("run1", {'loss-test':test_loss})
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

    epochs = 1000
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)

    writer.flush()
    writer.close()
    print("Done!")


if __name__ == '__main__':

    run(3, 1000, (8*8))

    print(0)
