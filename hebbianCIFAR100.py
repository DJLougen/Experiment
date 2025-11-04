import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# ---------------------------
# Cell Population Modulator
# ---------------------------
class CellPopMod(nn.Module):
    def __init__(self, channels, strength=1.0):
        super().__init__()
        self.strength = strength
        self.mask = nn.Parameter(torch.randn(1, channels, 1, 1))  # [1, C, 1, 1]

    def forward(self, x):
        return x * (1 + self.strength * torch.tanh(self.mask))

# ---------------------------
# Filters
# ---------------------------
def magno_filter():
    return nn.Sequential(
        nn.Conv2d(3, 32, 9, stride=4, padding=2), nn.ReLU(),
        nn.Conv2d(32, 64, 7, stride=2, padding=2), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1)
    )

def parvo_filter():
    return nn.Sequential(
        nn.Conv2d(3, 32, 5, padding=1), nn.ReLU(),
        nn.Conv2d(32, 32, 5, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 64, 2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1)
    )

def konio_filter():
    return nn.Sequential(
        nn.Conv2d(3, 16, 1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1)
    )

# ---------------------------
# DV Head (no backprop)
# ---------------------------
class DVLayer(nn.Module):
    def __init__(self, in_dim, out_dim, lr=0.01):
        super().__init__()
        self.weights = torch.zeros(out_dim, in_dim)
        self.lr = lr
        self.register_buffer('dv_weights', self.weights)

    def forward(self, x):
        return F.linear(x, self.dv_weights)

    def update(self, x, y):
        with torch.no_grad():
            y_onehot = F.one_hot(y, num_classes=self.dv_weights.size(0)).float()
            delta = torch.einsum('bi,bj->ij', y_onehot, x) / x.size(0)
            self.dv_weights.data = (1 - self.lr) * self.dv_weights + self.lr * delta

# ---------------------------
# DVX Model
# ---------------------------
class DVX(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()

        self.pop_mag = CellPopMod(3)
        self.pop_par = CellPopMod(3)
        self.pop_kon = CellPopMod(3)

        self.mag = magno_filter()
        self.par = parvo_filter()
        self.kon = konio_filter()

        self.dv = DVLayer(64 + 64 + 16, num_classes)  # Magno + Parvo + Konio outputs

    def forward(self, x):
        m = self.mag(self.pop_mag(x)).flatten(1)
        p = self.par(self.pop_par(x)).flatten(1)
        k = self.kon(self.pop_kon(x)).flatten(1)
        combined = torch.cat([m, p, k], dim=1)
        return self.dv(combined)

    def dv_update(self, x, y):
        m = self.mag(self.pop_mag(x)).flatten(1)
        p = self.par(self.pop_par(x)).flatten(1)
        k = self.kon(self.pop_kon(x)).flatten(1)
        combined = torch.cat([m, p, k], dim=1)
        self.dv.update(combined, y)

# ---------------------------
# CIFAR-100 Setup
# ---------------------------
transform_train = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(32, padding=4),
    T.ToTensor()
])

transform_test = T.Compose([
    T.ToTensor()
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
testset  = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
testloader  = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

# ---------------------------
# Training Loop (No Backprop)
# ---------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = DVX(num_classes=100).to(device)

for epoch in range(100):
    model.train()
    total_correct, total = 0, 0
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        total_correct += (preds.argmax(1) == y).sum().item()
        total += x.size(0)
        model.dv_update(x, y)

    train_acc = total_correct / total
    print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.4f}")

    # Evaluation
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            total_correct += (preds.argmax(1) == y).sum().item()

    test_acc = total_correct / len(testset)
    print(f"→ Test Acc: {test_acc:.4f}")
