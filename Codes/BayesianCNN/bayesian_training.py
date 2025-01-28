from __future__ import print_function
import os
import argparse
import torch
import numpy as np
import pandas as pd
import cv2 as cv
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config_bayesian as cfg
import metrics
import utils
from layers import BBB_Linear, BBB_Conv2d, BBB_LRT_Linear, BBB_LRT_Conv2d, FlattenLayer, ModuleWrapper

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_dir = '/vscratch/grp-danialfa/Azharul/BCNN/data/'
list_file = ['/vscratch/grp-danialfa/Azharul/BCNN/data/New_training_1']
csv_path = [os.path.join(dataset_dir, f'{file}.csv') for file in list_file]
df = pd.concat((pd.read_csv(file).assign(filename=file) for file in csv_path), ignore_index=True)
train_images = df['image_path'].apply(lambda x: cv.imread(os.path.join(dataset_dir, 'Image', x), 0))
train_scores = np.asarray(df['Strain_Energy'])

transform_split_mnist = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((168, 168)),
    transforms.ToTensor(),
])


class CustomDataset(Dataset):
    def __init__(self, images, scores, transform=None):
        self.images = images
        self.scores = scores.reshape(-1, 1)
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image = self.images[index]
        score = self.scores[index]
        if self.transform is not None:
            image = self.transform(image)
        score = torch.tensor(score).float()
        return image, score

train_ratio = 0.88
val_ratio = 0.10
test_ratio = 0.02
dataset = CustomDataset(train_images, train_scores, transform=transform_split_mnist)
total_size = len(dataset)
train_size = int(total_size * train_ratio)
val_size = int(total_size * val_ratio)
test_size = total_size - train_size - val_size
indices = list(range(total_size))
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)
batch_size = 16
shuffle_dataset = True
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_dataset)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
inputs = 1
outputs = 1

class BBBLeNetRegression(ModuleWrapper):
    def __init__(self, inputs, priors, layer_type='lrt', activation_type='softplus'):
        super(BBBLeNetRegression, self).__init__()
        self.layer_type = layer_type
        self.priors = priors
        if layer_type == 'lrt':
            BBBLinear = BBB_LRT_Linear
            BBBConv2d = BBB_LRT_Conv2d
        elif layer_type == 'bbb':
            BBBLinear = BBB_Linear
            BBBConv2d = BBB_Conv2d
        else:
            raise ValueError("Undefined layer_type")
        if activation_type == 'softplus':
            self.act = nn.Softplus
        elif activation_type == 'relu':
            self.act = nn.ReLU
        else:
            raise ValueError("Only softplus or relu supported")
        self.conv1 = BBBConv2d(inputs, 6, 5, padding=0, bias=True, priors=self.priors)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = BBBConv2d(6, 16, 5, padding=0, bias=True, priors=self.priors)
        self.act2 = self.act()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = FlattenLayer(16 * 39 * 39)
        self.fc1 = BBBLinear(16 * 39 * 39, 120, bias=True, priors=self.priors)
        self.act3 = self.act()
        self.fc2 = BBBLinear(120, 84, bias=True, priors=self.priors)
        self.act4 = self.act()
        self.regression_layer = BBBLinear(84, 1, bias=True, priors=self.priors)

class ELBO(nn.Module):
    def __init__(self, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size
    def forward(self, input, target, kl, beta):
        target = target.float()
        if target.dim() != input.dim():
            target = target.unsqueeze(1)
        return F.mse_loss(input, target, reduction='mean') * self.train_size + beta * kl

def calculate_rmse(outputs, targets):
    return torch.sqrt(F.mse_loss(outputs, targets))

def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    if isinstance(beta_type, float):
        return beta_type
    if beta_type == "Blundell":
        return 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs.')
        return min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        return 1 / m
    else:
        return 0

def logmeanexp(x, dim=None, keepdim=False):
    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim=1, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)

def train_model(net, optimizer, criterion, train_loader, num_ens=10, beta_type=0.1, epoch=None, num_epochs=None):
    net.train()
    training_loss = 0.0
    kl_list = []
    for i, (inputs_batch, targets) in enumerate(train_loader, 1):
        optimizer.zero_grad()
        inputs_batch, targets = inputs_batch.to(device), targets.to(device)
        outputs = torch.zeros(inputs_batch.shape[0], num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs_batch)
            kl += _kl
            outputs[:, j] = net_out.squeeze()
        outputs = outputs.unsqueeze(2)
        kl = kl / num_ens
        kl_list.append(kl.item())
        log_outputs = logmeanexp(outputs, dim=2)
        beta = get_beta(i - 1, len(train_loader), beta_type, epoch, num_epochs)
        loss = criterion(log_outputs, targets, kl, beta)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
    return training_loss / len(train_loader), np.mean(kl_list)

def validate_model(net, criterion, valid_loader, num_ens=10, beta_type=0.1, epoch=None, num_epochs=None):
    net.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for i, (inputs_batch, targets) in enumerate(valid_loader):
            inputs_batch, targets = inputs_batch.to(device), targets.to(device)
            outputs = torch.zeros(inputs_batch.shape[0], num_ens).to(device)
            kl = 0.0
            for j in range(num_ens):
                net_out, _kl = net(inputs_batch)
                kl += _kl
                outputs[:, j] = net_out.squeeze()
            outputs = outputs.unsqueeze(2)
            kl = kl / num_ens
            log_outputs = logmeanexp(outputs, dim=2)
            beta = get_beta(i - 1, len(valid_loader), beta_type, epoch, num_epochs)
            valid_loss += criterion(log_outputs, targets, kl, beta).item()
    return valid_loss / len(valid_loader)

priors = {
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),
    'posterior_rho_initial': (-3, 0.1)
}

def run(dataset, net_type, train_loader, valid_loader, test_loader, inputs, outputs, priors):
    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens
    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    beta_type = cfg.beta_type
    net = BBBLeNetRegression(inputs, priors, layer_type='lrt', activation_type='softplus').to(device)
    ckpt_dir = '/vscratch/grp-danialfa/Azharul/BCNN/checkpoints/MNIST'
    ckpt_name = '/vscratch/grp-danialfa/Azharul/BCNN/checkpoints/MNIST/Lenet_bayesian_regression.pt'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    criterion = ELBO(len(train_loader.dataset)).to(device)
    optimizer = Adam(net.parameters(), lr=lr_start)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.8, verbose=True)
    valid_loss_min = np.Inf
    for epoch in range(n_epochs):
        train_loss, train_kl = train_model(net, optimizer, criterion, train_loader, num_ens=train_ens,
                                           beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
        valid_loss = validate_model(net, criterion, val_loader, num_ens=valid_ens, beta_type=beta_type,
                                    epoch=epoch, num_epochs=n_epochs)
        lr_sched.step(valid_loss)
        print('Epoch: {}\tTraining Loss: {:.4f}\tValidation Loss: {:.4f}\tTrain KL Divergence: {:.4f}'.format(
            epoch, train_loss, valid_loss, train_kl))
        uncertainty_values = net.conv1.W_sigma
        uncertainty_list = uncertainty_values.detach().cpu().numpy().tolist()
        file_path = "/vscratch/grp-danialfa/Azharul/BCNN/conv1_sigma.txt"
        with open(file_path, "a") as file:
            file.write(f"Epoch {epoch}:\n")
            for value in uncertainty_list:
                file.write(str(value) + "\n")
    torch.save(net.state_dict(), ckpt_name)
    net.load_state_dict(torch.load(ckpt_name))
    return net

net = run('RegressionDataset', 'bayesian_regression', train_loader, val_loader, test_loader, inputs, outputs, priors)

def test_bayesian_model(net, test_loader, num_samples=100):
    net.eval()
    rmse_values = []
    r2_values = []
    std_values = []
    with torch.no_grad():
        for _ in range(num_samples):
            predictions = []
            targets_all = []
            stds = []
            for inputs_batch, targets_batch in test_loader:
                inputs_batch = inputs_batch.to(device)
                targets_batch = targets_batch.to(device)
                outputs = []
                for i in range(num_samples):
                    net_out, _ = net(inputs_batch)
                    outputs.append(net_out.squeeze().cpu().numpy())
                outputs = np.array(outputs)
                mean_prediction = np.mean(outputs, axis=0)
                std_prediction = np.std(outputs, axis=0)
                predictions.extend(mean_prediction)
                targets_all.extend(targets_batch.cpu().numpy())
                stds.extend(std_prediction)
            rmse = calculate_rmse(torch.tensor(predictions), torch.tensor(targets_all))
            rmse_values.append(rmse.item())
            r2 = r2_score(targets_all, predictions)
            r2_values.append(r2)
            std_values.append(np.mean(stds))
    return rmse_values, r2_values, std_values

num_samples = 50
rmse_values, r2_values, std_values = test_bayesian_model(net, test_loader, num_samples)
mean_rmse = np.mean(rmse_values)
mean_r2 = np.mean(r2_values)
print(f"Mean RMSE over {num_samples} runs: {mean_rmse:.4f}")
print(f"Mean R^2 over {num_samples} runs: {mean_r2:.4f}")
