from __future__ import print_function
import argparse
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import torchvision
from torch.nn import functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import config_bayesian as cfg
from torchvision.transforms import ToTensor
import torch.nn as nn
from layers import BBB_Linear, BBB_Conv2d
from layers import BBB_LRT_Linear, BBB_LRT_Conv2d
from layers import FlattenLayer, ModuleWrapper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_user_image(image_path):
    img = Image.open(image_path).convert('L')
    img = ToTensor()(img)
    transform_pipeline = transforms.Compose([
        transforms.Resize((168, 168)),
        transforms.ToTensor(),
    ])
    img = transform_pipeline(img)
    return img.to(device)

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

def get_uncertainty_per_image(model, input_image, T=15, normalized=False):
    input_image = input_image.unsqueeze(0)
    input_images = input_image.repeat(T, 1, 1, 1)
    net_out, _ = model(input_images)
    pred = torch.mean(net_out, dim=0).cpu().detach().numpy()
    p_hat = net_out.detach().cpu().numpy()
    epistemic = np.var(p_hat, axis=0)
    aleatoric = np.mean(np.abs(p_hat - pred), axis=0)
    return pred, epistemic, aleatoric, p_hat

def visualize_uncertainty(model, input_image, T=15, normalized=False):
    pred, epistemic, aleatoric, individual_predictions = get_uncertainty_per_image(
        model, input_image, T=T, normalized=normalized
    )
    individual_predictions1 = individual_predictions.flatten()
    print(f"Predicted Value: {np.mean(pred)}")
    print("Predicted Values:", pred)
    print("Epistemic Uncertainty:", epistemic)
    print("Aleatoric Uncertainty:", aleatoric)
    print(individual_predictions1)
    img = transforms.ToPILImage()(input_image.cpu().squeeze())
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig('image.png')
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=individual_predictions1, shade=True)
    plt.xlabel("Predicted Value")
    plt.ylabel("Density Estimate")
    plt.title("Probability Density Estimate for Predicted Values at Each Iteration")
    plt.savefig('image_plot.png')
    density_plot = sns.kdeplot(individual_predictions1)
    x_vals, y_vals = density_plot.get_lines()[0].get_data()
    max_density_index = np.argmax(y_vals)
    max_density_x = x_vals[max_density_index]
    print(f"Highest Density Point - X: {max_density_x}")

def run(net_type, weight_path, image_path):
    layer_type = cfg.layer_type
    activation_type = cfg.activation_type
    net = BBBLeNetRegression(inputs=1, priors=None, layer_type=layer_type, activation_type=activation_type)
    net.load_state_dict(torch.load(weight_path))
    net.train()
    net.to(device)
    input_image = load_user_image(image_path)
    visualize_uncertainty(net, input_image, T=40, normalized=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Uncertainty Estimation on MNIST")
    parser.add_argument('--net_type', default='lenet', type=str, help='model')
    parser.add_argument('--weights_path', default='/vscratch/grp-danialfa/Azharul/BCNN/checkpoints/MNIST/Lenet_bayesian_regression.pt', type=str, help='weights for model')
    parser.add_argument('--image_path', default='/vscratch/grp-danialfa/Azharul/BCNN/test/22.1.png', type=str, help='path to the input image')
    args = parser.parse_args()
    run(args.net_type, args.weights_path, args.image_path)
