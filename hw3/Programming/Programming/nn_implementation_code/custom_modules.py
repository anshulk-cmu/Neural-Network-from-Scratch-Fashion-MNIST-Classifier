import torch
from torch.nn import Module, Parameter

from custom_functions import IdentityFunction, LinearFunction, CrossEntropyFunction, SigmoidFunction


class Identity(Module):
    def __init__(self):
        # An identity layer does nothing
        super().__init__()

    def forward(self, inp):
        # An identity layer just returns whatever it gets as input.
        return IdentityFunction.apply(inp)


class Linear(Module):
    def __init__(self, input_units, output_units):
        super().__init__()
        # initialize weights with small random numbers from normal distribution
        weight = torch.rand(output_units, input_units) * 0.01
        bias = torch.zeros(output_units)
        self.weight = Parameter(weight)
        self.bias = Parameter(bias)

    def forward(self, inp):
        return LinearFunction.apply(inp, self.weight, self.bias)


class CrossEntropyLoss(Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, activations, target):
        return CrossEntropyFunction.apply(activations, target)


class Sigmoid(Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, input):
        return SigmoidFunction.apply(input)


if __name__ == "__main__":
    from collections import OrderedDict
    from torch.autograd import gradcheck

    num = 4
    inp = 3
    out = 2

    x = torch.rand((num, inp), requires_grad=True).double()
    weight = torch.rand((out, inp), requires_grad=True).double()
    bias = torch.rand(out, requires_grad=True).double()
    linear = Linear(3, 2).double()
    state_dict = OrderedDict([("weight", weight), ("bias", bias)])
    linear.load_state_dict(state_dict)
    assert gradcheck(linear, (x,))

    torch_linear = torch.nn.Linear(3, 2).double()
    our_linear = Linear(3, 2).double()

    state_dict = OrderedDict([("weight", weight), ("bias", bias)])
    print(torch_linear.load_state_dict(state_dict))
    print(our_linear.load_state_dict(state_dict))

    assert (our_linear(x) == torch_linear(x)).all()

    print("Our fully connected layer has exactly the same interface as `torch.nn.Linear`")
