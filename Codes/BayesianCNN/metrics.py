import numpy as np
import torch.nn.functional as F
from torch import nn
import torch

class ELBO(nn.Module):
    def __init__(self, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size

    def forward(self, input, target_mu, target_sigma, kl, beta):
        # Use mean squared error (MSE) loss for regression task
        reconstruction_loss = F.mse_loss(input, target_mu, reduction='mean')
        return reconstruction_loss * self.train_size + beta * kl

def acc(outputs, targets):
    # This function is used for classification tasks, modify or omit it for regression.
    # For regression, you typically evaluate using metrics like RMSE or R2 score.
    return np.mean(outputs.cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())

def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta
