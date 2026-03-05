from collections import Counter

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from custom_modules import Linear, CrossEntropyLoss, Sigmoid

class FashionMNISTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = Linear(784, 256)
        self.sigmoid = Sigmoid()
        self.lin2 = Linear(256, 10)


    def forward(self, x):
        # Ensure batch dimension exists, then flatten to (batch, 784)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x_flat = x.view(x.size(0), -1)
        a = self.lin1(x_flat)
        z = self.sigmoid(a)
        logits = self.lin2(z)
        return logits


def q1_to_q6(model: FashionMNISTModel, trainset: torchvision.datasets, testset: torchvision.datasets, lr=0.01, epohs=15, device= torch.device):
    """
    Return:
        (Q1: Float, Q2: Float, Q3: Integer, Q4: List of floats, 
        Q5: List of floats (rounded to 4 d.p.), Q6: List of floats (rounded to 4 d.p.))
    """
    # TODO: Initialize initial weights and bias for linear layers

    # Initialized optimizer for SGD
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    # Intialized loss metric
    loss_func = CrossEntropyLoss()
    optim.zero_grad()

    # TODO: Implement rest of function
    pass 

def q7(model: FashionMNISTModel, trainset: torchvision.datasets, testset: torchvision.datasets, lr=0.01, epohs=50, device= torch.device):
    """
    Return:
        (Training loss: Float, Test Accuracy: Float)
    """
    # TODO: Implement function 
    pass


if __name__ == "__main__":
    trainset = torchvision.datasets.FashionMNIST(root='./', train=True,
                                                 download=True, transform=transforms.ToTensor())

    testset = torchvision.datasets.FashionMNIST(root='./', train=False,
                                                download=True, transform=transforms.ToTensor())

    weights = torch.load("weights.pt")
    print(weights)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ...

    Q1, Q2, Q3, Q4, Q5, Q6 = q1_to_q6(...)

    # Note that batch size of data changed

    # ...

    Q7_loss, Q7_accuracy = q7(...)


