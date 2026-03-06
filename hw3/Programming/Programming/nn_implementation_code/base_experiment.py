import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    # Load initial weights
    model.load_state_dict(torch.load("weights.pt"))
    model.to(device)

    # Initialized optimizer for SGD
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    # Intialized loss metric
    loss_func = CrossEntropyLoss()

    # DataLoaders — batch_size=1, no shuffle
    train_loader = DataLoader(trainset, batch_size=1, shuffle=False)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False)

    # Placeholders for Q1-Q6 answers
    Q1 = None  # a_10 for first data point, first epoch
    Q2 = None  # z_20 for first data point, first epoch
    Q3 = None  # predicted class for first data point, first epoch
    Q4 = None  # bias of second layer after epoch 3
    Q5 = []    # test loss at end of each epoch
    Q6 = []    # test accuracy at end of each epoch

    for epoch in range(epohs):
        # --- Training loop ---
        for i, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epohs}")):
            x, y = x.to(device), y.to(device)

            optim.zero_grad()

            # Forward pass — capture intermediates for Q1/Q2/Q3 on first data point of first epoch
            if epoch == 0 and i == 0:
                x_flat = x.view(x.size(0), -1)
                a = model.lin1(x_flat)
                z = model.sigmoid(a)
                logits = model.lin2(z)

                Q1 = round(a[0, 9].item(), 4)   # a_10 (0-indexed: index 9)
                Q2 = round(z[0, 19].item(), 4)  # z_20 (0-indexed: index 19)
                Q3 = int(logits[0].argmax().item())  # predicted class
            else:
                logits = model(x)

            loss = loss_func(logits, y)
            loss.backward()
            optim.step()

        # Q4: bias of second layer after epoch 3
        if epoch == 2:
            Q4 = [round(v.item(), 4) for v in model.lin2.bias.data]

        # --- Compute training loss (fresh forward pass over full training set) ---
        train_loss_total = 0.0
        with torch.no_grad():
            for x_tr, y_tr in train_loader:
                x_tr, y_tr = x_tr.to(device), y_tr.to(device)
                logits_tr = model(x_tr)

                logits_2d = logits_tr.view(logits_tr.size(0), -1)
                max_logits = logits_2d.max(dim=1, keepdim=True).values
                shifted = logits_2d - max_logits
                log_sum_exp = torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))
                log_softmax = shifted - log_sum_exp
                bs = logits_2d.shape[0]
                loss_per_sample = -log_softmax[torch.arange(bs), y_tr]
                train_loss_total += loss_per_sample.sum().item()

        avg_train_loss = train_loss_total / len(trainset)

        # --- Compute test loss and accuracy (fresh forward pass over full test set) ---
        test_loss_total = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(device), y_test.to(device)
                logits_test = model(x_test)

                logits_2d = logits_test.view(logits_test.size(0), -1)
                max_logits = logits_2d.max(dim=1, keepdim=True).values
                shifted = logits_2d - max_logits
                log_sum_exp = torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))
                log_softmax = shifted - log_sum_exp
                bs = logits_2d.shape[0]
                loss_per_sample = -log_softmax[torch.arange(bs), y_test]
                test_loss_total += loss_per_sample.sum().item()

                preds = logits_test.argmax(dim=1)
                correct += (preds == y_test).sum().item()
                total += y_test.size(0)

        avg_test_loss = test_loss_total / total
        test_accuracy = correct / total

        Q5.append(round(avg_test_loss, 4))
        Q6.append(round(test_accuracy, 4))

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")

    return Q1, Q2, Q3, Q4, Q5, Q6

def q7(model: FashionMNISTModel, trainset: torchvision.datasets, testset: torchvision.datasets, lr=0.01, epohs=50, device= torch.device):
    """
    Return:
        (Training loss: Float, Test Accuracy: Float)
    """
    # Fresh start — reload initial weights
    model.load_state_dict(torch.load("weights.pt"))
    model.to(device)

    optim = torch.optim.SGD(model.parameters(), lr=lr)
    loss_func = CrossEntropyLoss()

    # DataLoaders — batch_size=5, no shuffle
    train_loader = DataLoader(trainset, batch_size=5, shuffle=False)
    test_loader = DataLoader(testset, batch_size=5, shuffle=False)

    for epoch in range(epohs):
        # --- Training loop ---
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epohs}"):
            x, y = x.to(device), y.to(device)

            optim.zero_grad()
            logits = model(x)
            loss = loss_func(logits, y)
            loss.backward()
            optim.step()

        print(f"Epoch {epoch+1}/{epohs} complete")

    # --- Final training loss (fresh forward pass over full training set) ---
    train_loss_total = 0.0
    with torch.no_grad():
        for x_tr, y_tr in train_loader:
            x_tr, y_tr = x_tr.to(device), y_tr.to(device)
            logits_tr = model(x_tr)

            logits_2d = logits_tr.view(logits_tr.size(0), -1)
            max_logits = logits_2d.max(dim=1, keepdim=True).values
            shifted = logits_2d - max_logits
            log_sum_exp = torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))
            log_softmax = shifted - log_sum_exp
            bs = logits_2d.shape[0]
            loss_per_sample = -log_softmax[torch.arange(bs), y_tr]
            train_loss_total += loss_per_sample.sum().item()

    avg_train_loss = round(train_loss_total / len(trainset), 4)

    # --- Final test accuracy ---
    correct = 0
    total = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            logits_test = model(x_test)
            preds = logits_test.argmax(dim=1)
            correct += (preds == y_test).sum().item()
            total += y_test.size(0)

    test_accuracy = round(correct / total, 4)

    print(f"Final Training Loss: {avg_train_loss}, Final Test Accuracy: {test_accuracy}")
    return avg_train_loss, test_accuracy


def q8(model: FashionMNISTModel, trainset: torchvision.datasets, testset: torchvision.datasets, device=torch.device):
    """
    Generate confusion matrices for training and test sets.
    Uses the model in its current state (should be called after Q7 training).
    """
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_loader = DataLoader(trainset, batch_size=100, shuffle=False)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False)

    # Collect predictions for training set
    all_train_true, all_train_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(train_loader, desc="Train confusion matrix"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_train_true.extend(y.cpu().numpy())
            all_train_pred.extend(preds.cpu().numpy())

    # Collect predictions for test set
    all_test_true, all_test_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Test confusion matrix"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_test_true.extend(y.cpu().numpy())
            all_test_pred.extend(preds.cpu().numpy())

    # Plot confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Training confusion matrix
    cm_train = confusion_matrix(all_train_true, all_train_pred)
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=class_names)
    disp_train.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title('Training Set Confusion Matrix', fontsize=14)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')

    # Test confusion matrix
    cm_test = confusion_matrix(all_test_true, all_test_pred)
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=class_names)
    disp_test.plot(ax=axes[1], cmap='Blues', values_format='d')
    axes[1].set_title('Test Set Confusion Matrix', fontsize=14)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('q8_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved confusion matrices to q8_confusion_matrices.png")


def q9(model: FashionMNISTModel, testset: torchvision.datasets, device=torch.device):
    """
    For each class (0-9), find and display the first misclassified test image.
    """
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Track first misclassification per class: {class_id: (index, image, true_label, pred_label)}
    misclassified = {}
    classes_found = set()

    with torch.no_grad():
        for i in range(len(testset)):
            x, y = testset[i]
            x_dev = x.to(device)
            logits = model(x_dev.unsqueeze(0))
            pred = logits.argmax(dim=1).item()

            if pred != y and y not in classes_found:
                misclassified[y] = (i, x.squeeze().cpu().numpy(), y, pred)
                classes_found.add(y)

            if len(classes_found) == 10:
                break

    # Plot 2 rows x 5 cols
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for cls_id in range(10):
        row, col = cls_id // 5, cls_id % 5
        ax = axes[row][col]
        if cls_id in misclassified:
            idx, img, true_lbl, pred_lbl = misclassified[cls_id]
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Idx {idx}\nTrue: {class_names[true_lbl]}\nPred: {class_names[pred_lbl]}',
                        fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No errors', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{class_names[cls_id]}\n(all correct)', fontsize=9)
        ax.axis('off')

    plt.suptitle('Q9: First Misclassified Test Image per Class', fontsize=14)
    plt.tight_layout()
    plt.savefig('q9_first_misclassified.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved first misclassified images to q9_first_misclassified.png")


def compute_avg_loss(model, dataset, batch_size, device):
    """Compute average cross-entropy loss over an entire dataset (no grad)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            logits_2d = logits.view(logits.size(0), -1)
            max_logits = logits_2d.max(dim=1, keepdim=True).values
            shifted = logits_2d - max_logits
            log_sum_exp = torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))
            log_softmax = shifted - log_sum_exp
            bs = logits_2d.shape[0]
            loss_per_sample = -log_softmax[torch.arange(bs), y]
            total_loss += loss_per_sample.sum().item()
    return total_loss / len(dataset)


def q10(trainset: torchvision.datasets, testset: torchvision.datasets, lr=0.01, epochs=50, device=torch.device):
    """
    Train models with batch sizes [10, 50, 100] for 50 epochs each.
    Return dict: {batch_size: {'train_losses': [...], 'test_losses': [...]}}
    """
    batch_sizes = [10, 50, 100]
    results = {}

    for bs in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Training with batch_size={bs} for {epochs} epochs")
        print(f"{'='*60}")

        # Fresh model with initial weights
        model = FashionMNISTModel().to(device)
        model.load_state_dict(torch.load("weights.pt"))

        optim = torch.optim.SGD(model.parameters(), lr=lr)
        loss_func = CrossEntropyLoss()
        train_loader = DataLoader(trainset, batch_size=bs, shuffle=False)

        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            # --- Training loop ---
            for x, y in tqdm(train_loader, desc=f"BS={bs} Epoch {epoch+1}/{epochs}"):
                x, y = x.to(device), y.to(device)
                optim.zero_grad()
                logits = model(x)
                loss = loss_func(logits, y)
                loss.backward()
                optim.step()

            # --- Epoch-end evaluation ---
            avg_train_loss = compute_avg_loss(model, trainset, bs, device)
            avg_test_loss = compute_avg_loss(model, testset, bs, device)
            train_losses.append(round(avg_train_loss, 4))
            test_losses.append(round(avg_test_loss, 4))

            print(f"  Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}")

        results[bs] = {'train_losses': train_losses, 'test_losses': test_losses}

    # --- Plot: Training Loss ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    epochs_range = list(range(1, epochs + 1))

    for bs in batch_sizes:
        axes[0].plot(epochs_range, results[bs]['train_losses'], label=f'Batch Size = {bs}')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss vs Epoch', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # --- Plot: Test Loss ---
    for bs in batch_sizes:
        axes[1].plot(epochs_range, results[bs]['test_losses'], label=f'Batch Size = {bs}')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Test Loss vs Epoch', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('q10_batch_size_experiments.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved batch size experiment plots to q10_batch_size_experiments.png")

    return results


def q13(trainset: torchvision.datasets, testset: torchvision.datasets, epochs=50, device=torch.device):
    """
    Q13: Hyperparameter experiment — varying Learning Rate.

    Hyperparameter varied: Learning Rate
    Values tested: [0.001, 0.01, 0.1]
    Fixed settings:
        - Batch size: 5
        - Epochs: 50
        - Hidden layer width: 256
        - Weight initialization: weights.pt (identical for all runs)
        - Optimizer: SGD
        - No data shuffling
    """
    learning_rates = [0.001, 0.01, 0.1]
    batch_size = 5
    results = {}

    for lr in learning_rates:
        print(f"\n{'='*60}")
        print(f"Training with lr={lr}, batch_size={batch_size}, epochs={epochs}")
        print(f"{'='*60}")

        # Fresh model with identical initial weights
        model = FashionMNISTModel().to(device)
        model.load_state_dict(torch.load("weights.pt"))

        optim = torch.optim.SGD(model.parameters(), lr=lr)
        loss_func = CrossEntropyLoss()
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            # --- Training loop ---
            for x, y in tqdm(train_loader, desc=f"LR={lr} Epoch {epoch+1}/{epochs}"):
                x, y = x.to(device), y.to(device)
                optim.zero_grad()
                logits = model(x)
                loss = loss_func(logits, y)
                loss.backward()
                optim.step()

            # --- Epoch-end evaluation ---
            avg_train_loss = compute_avg_loss(model, trainset, batch_size, device)
            avg_test_loss = compute_avg_loss(model, testset, batch_size, device)
            train_losses.append(round(avg_train_loss, 4))
            test_losses.append(round(avg_test_loss, 4))

            print(f"  Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}")

        results[lr] = {'train_losses': train_losses, 'test_losses': test_losses}

    # --- Plot: Training Loss ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    epochs_range = list(range(1, epochs + 1))

    for lr in learning_rates:
        axes[0].plot(epochs_range, results[lr]['train_losses'], label=f'LR = {lr}')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss vs Epoch (Varying Learning Rate)', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # --- Plot: Test Loss ---
    for lr in learning_rates:
        axes[1].plot(epochs_range, results[lr]['test_losses'], label=f'LR = {lr}')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Test Loss vs Epoch (Varying Learning Rate)', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('q13_learning_rate_experiment.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved learning rate experiment plots to q13_learning_rate_experiment.png")

    return results


if __name__ == "__main__":
    trainset = torchvision.datasets.FashionMNIST(root='./', train=True,
                                                 download=True, transform=transforms.ToTensor())

    testset = torchvision.datasets.FashionMNIST(root='./', train=False,
                                                download=True, transform=transforms.ToTensor())

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ===== Q1–Q6: batch_size=1, 15 epochs =====
    model = FashionMNISTModel().to(device)
    Q1, Q2, Q3, Q4, Q5, Q6 = q1_to_q6(model, trainset, testset, lr=0.01, epohs=15, device=device)

    print("\n" + "="*60)
    print("ANSWERS Q1–Q6")
    print("="*60)
    print(f"Q1 (a_10):           {Q1}")
    print(f"Q2 (z_20):           {Q2}")
    print(f"Q3 (predicted class): {Q3}")
    print(f"Q4 (beta biases):    {Q4}")
    print(f"Q5 (test losses):    {Q5}")
    print(f"Q6 (test accuracy):  {Q6}")

    # ===== Q7: batch_size=5, 50 epochs =====
    model = FashionMNISTModel().to(device)
    Q7_loss, Q7_accuracy = q7(model, trainset, testset, lr=0.01, epohs=50, device=device)

    print("\n" + "="*60)
    print("ANSWERS Q7")
    print("="*60)
    print(f"Q7 Training Loss:    {Q7_loss}")
    print(f"Q7 Test Accuracy:    {Q7_accuracy}")

    # ===== Q8: Confusion matrices (uses Q7-trained model) =====
    q8(model, trainset, testset, device=device)

    # ===== Q9: First misclassified test image per class =====
    q9(model, testset, device=device)

    # ===== Q10: Batch size experiments (10, 50, 100) =====
    q10_results = q10(trainset, testset, lr=0.01, epochs=50, device=device)

    # ===== Q13: Learning rate experiment =====
    q13_results = q13(trainset, testset, epochs=50, device=device)
