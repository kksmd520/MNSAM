import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from models import *
from sam import SAM,DSAM,enable_running_stats,disable_running_stats
import torchvision.models as models
from torchvision.models import resnet18
# 加载MNIST数据集
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
#
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 加载cifar10数据集

transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4940607, 0.4850613, 0.45037037], [0.20085774, 0.19870903, 0.20153421])
        ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4940607, 0.4850613, 0.45037037], [0.20085774, 0.19870903, 0.20153421])
    ])
train_dataset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True)
test_dataset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=100, shuffle=False)

class_names = [str(i) for i in range(10)]

# 可视化部分训练样本

# 定义模型


import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # 定义卷积层
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 定义全连接层
        self.fc1 = nn.Linear(28 * 28, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # 对输入做维度变换 [batch, 1, 28, 28]
        x = x.view(-1, 1, 28, 28)

        # 卷积和池化层
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)

        # 展平
        x = x.view(-1, 28 * 28)

        # 全连接层
        x = F.selu(self.fc1(x))
        x = self.fc2(x)

        return x
num_classes = 10
net = ResNet18(num_classes)
model=net
# model = resnet18(num_classes)
# model = MNISTNet()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# # optimizer = DSAM(model.parameters(), torch.optim.SGD, rho=0.01, adaptive=False, lr=0.01,
# #                         momentum=0.9,  nesterov=True)
#
# # 训练模型
# num_epochs = 5
# for epoch in range(num_epochs):
#     for batch_idx, (inputs, targets) in enumerate(train_loader):
#         inputs_dev, targets_dev = inputs.to(device), targets.to(device)
#         if optimizer.__class__.__name__ == 'SAM' or optimizer.__class__.__name__ == 'DSAM':
#             enable_running_stats(model)
#             outputs = model(inputs_dev)  # 通过前向传播获取网络输出
#             loss = criterion(outputs, targets_dev)  # 计算损失
#             loss.backward()  # 自动计算梯度
#             optimizer.first_step(zero_grad=True)
#             # second forward-backward step
#             disable_running_stats(model)
#             criterion(model(inputs_dev), targets_dev).backward()
#             optimizer.second_step(zero_grad=True)
#         else:
#             optimizer.zero_grad()
#             outputs = model(inputs_dev)
#             loss = criterion(outputs, targets_dev)
#             loss.backward()
#             optimizer.step()
#
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
#
# # 在测试集上评估模型
# model.eval()
# test_loss = 0
# correct = 0
# with torch.no_grad():
#     for batch_idx, (inputs, targets) in enumerate(test_loader):
#         inputs_dev, targets_dev = inputs.to(device), targets.to(device)
#         outputs = model(inputs_dev)
#         test_loss += criterion(outputs, targets_dev).item()
#         _, predicted = torch.max(outputs.data, 1)
#         correct += (predicted == targets_dev).sum().item()
#
# test_loss /= len(test_loader)
# test_acc = correct / len(test_dataset)
# print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# 保存训练好的模型
torch.save(model.state_dict(), "trained_model.pth")

# def generate_random_direction(model):
#     list_of_weights = []
#     for weight in model.parameters():
#         list_of_weights.append(torch.randn_like(weight, device=device))  # Ensure the randomness is on the same device
#     return list_of_weights
def generate_random_direction(model):
    list_of_weights = []
    for weight in model.parameters():
        random_direction = weight * (1 + torch.randn_like(weight))
        list_of_weights.append(random_direction)
    return list_of_weights

bound = -2
alphas = np.linspace(-bound, bound, 5)
betas = np.linspace(-bound, bound, 5)
v_1 = generate_random_direction(model)
v_2 = generate_random_direction(model)
losses = np.zeros((len(alphas), len(betas)))

for i_alpha, alpha in tqdm(enumerate(alphas)):
    for i_beta, beta in tqdm(enumerate(betas), leave=False):
        # Load the model state dict for each iteration
        model_state = torch.load("trained_model.pth", map_location=device)
        model.load_state_dict(model_state)

        new_weights = []
        for weight, v_1_weight, v_2_weight in zip(model.parameters(), v_1, v_2):
            new_weight = weight.data + alpha * v_1_weight + beta * v_2_weight
            new_weights.append(new_weight)

        with torch.no_grad():
            for i, param in enumerate(model.parameters()):
                param.data.copy_(new_weights[i])

        test_loss = 0
        correct = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)
        losses[i_alpha, i_beta] = test_loss

# Transform losses into a format suitable for Plotly
# z_data = losses  # Transpose losses to match Plotly's axis expectations
# x, y = np.meshgrid(alphas, betas)
# fig = go.Figure(data=[go.Surface(z=z_data, x=x.ravel(), y=y.ravel())])
# fig.update_layout(title='2D Loss Surface Approximation', autosize=False,
#                   width=700, height=700,
#                   margin=dict(l=65, r=50, b=65, t=90))
# fig.show()
z_data = losses
z = losses
sh_0, sh_1 = z.shape
x, y = alphas, betas
fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
fig.update_layout(title='2d Loss surface approximation', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()
# 保存为html