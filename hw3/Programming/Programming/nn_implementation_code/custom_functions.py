import torch

from torch.autograd import Function


class IdentityFunction(Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(x):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return x

    @staticmethod
    def backward(grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        return grad_output


class SigmoidFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = 1.0 / (1.0 + torch.exp(-input))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        return grad_output * output * (1.0 - output)


class LinearFunction(Function):
    @staticmethod
    def forward(ctx, inp, weight, bias):
        ctx.save_for_backward(inp, weight, bias)
        output = inp @ weight.t() + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, bias = ctx.saved_tensors

        # Ensure 2D for matrix math (handle batch_size=1 case)
        if grad_output.dim() == 1:
            grad_output_2d = grad_output.unsqueeze(0)
            inp_2d = inp.unsqueeze(0)
        else:
            grad_output_2d = grad_output
            inp_2d = inp

        grad_inp = grad_output_2d @ weight          # (batch, in_features)
        grad_weight = grad_output_2d.t() @ inp_2d   # (out_features, in_features)
        grad_bias = grad_output_2d.sum(dim=0)        # (out_features,)

        # Restore original shape for grad_inp
        if inp.dim() == 1:
            grad_inp = grad_inp.squeeze(0)

        return grad_inp, grad_weight, grad_bias


class CrossEntropyFunction(Function):
    @staticmethod
    def forward(ctx, logits, target):
        # Ensure 2D: (batch, num_classes)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if target.dim() == 0:
            target = target.unsqueeze(0)

        # (a) Subtract max per sample for numerical stability
        max_logits = logits.max(dim=1, keepdim=True).values
        shifted = logits - max_logits

        # (b) Log-softmax: log_softmax = shifted - log(sum(exp(shifted)))
        log_sum_exp = torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))
        log_softmax = shifted - log_sum_exp

        # Softmax probabilities (needed for backward)
        softmax_probs = torch.exp(log_softmax)

        # (c) Pick log_softmax at target indices for per-sample loss
        batch_size = logits.shape[0]
        loss_per_sample = -log_softmax[torch.arange(batch_size), target]

        # (d) Mean loss
        loss = loss_per_sample.mean()

        # Save for backward
        ctx.save_for_backward(softmax_probs)
        ctx.target = target
        ctx.batch_size = batch_size

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (softmax_probs,) = ctx.saved_tensors
        target = ctx.target
        batch_size = ctx.batch_size

        # One-hot encoding of target
        one_hot = torch.zeros_like(softmax_probs)
        one_hot[torch.arange(batch_size), target] = 1.0

        # Gradient: (softmax - one_hot) / batch_size * grad_output
        grad_logits = (softmax_probs - one_hot) / batch_size * grad_output

        return grad_logits, None


if __name__ == "__main__":
    from torch.autograd import gradcheck

    num = 4
    inp = 3

    x = torch.rand((num, inp), requires_grad=True).double()

    sigmoid = SigmoidFunction.apply

    assert gradcheck(sigmoid, x)
    print("Backward pass for sigmoid function is implemented correctly")

    out = 2

    x = torch.rand((num, inp), requires_grad=True).double()
    weight = torch.rand((out, inp), requires_grad=True).double()
    bias = torch.rand(out, requires_grad=True).double()

    linear = LinearFunction.apply
    assert gradcheck(linear, (x, weight, bias))
    print("Backward pass for linear function is implemented correctly")

    activations = torch.rand((15, 10), requires_grad=True).double()
    target = torch.randint(10, (15,))
    crossentropy = CrossEntropyFunction.apply
    assert gradcheck(crossentropy, (activations, target))
    print("Backward pass for crossentropy function is implemented correctly")
