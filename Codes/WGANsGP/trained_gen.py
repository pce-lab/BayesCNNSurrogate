import torch
import torchvision
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import zipfile
from zipfile import ZipFile
import os
#from google.colab import drive
import matplotlib.pyplot as plt


class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels of the output image, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim, im_chan, hidden_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
      
        # Build the neural network
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(1024, hidden_dim*8, kernel_size=4, stride=2, padding = 1),
            nn.BatchNorm2d(hidden_dim*8),
            nn.ReLU(inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(hidden_dim*8, hidden_dim*4, kernel_size=4, stride=2, padding = 1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=2, padding = 1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(hidden_dim*2, hidden_dim, kernel_size=4, stride=2, padding = 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            #nn.ConvTranspose2d(64*8, 64*4, kernel_size=4, stride=2, padding = 1),
            #nn.BatchNorm2d(64*4),
            #nn.ReLU(inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(hidden_dim, im_chan, kernel_size=4, stride=2, padding = 1),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
              
        )

    def forward(self, x):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''

        x = self.gen(x)
        
        return x

def get_noise(batch_size, z_dim, device='cuda'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
      n_samples: the number of samples to generate, a scalar
      z_dim: the dimension of the noise vector, a scalar
      device: the device type
    '''
    m = nn.Linear(z_dim, 1024*(z_dim**2)).to("cuda")
    input = torch.randn(batch_size, z_dim, device=device).to("cuda")
    x  = m(input)
    x = x.view(x.size(0), 1024, z_dim, z_dim)
    return x
  










path = '/panasas/scratch/grp-danialfa/shayanbh/scalable_test4/model_new1.pth'
gen = torch.load(path)
print(gen.eval())

z_dim = 40
#lin = nn.Linear(z_dim, 1024*(z_dim**2))

#z_dim = 3

for i in range(1,21):
    fake_noise_3 = get_noise(1, z_dim, device='cuda')
    print(fake_noise_3.size())

#x = lin(fake_noise_3)
#x = x.view(x.size(0), 1024, z_dim, z_dim)
#print(x.size())
    fake_3 = gen(fake_noise_3)
    save_image(fake_3, "fake_trained_" + str(i) + ".png")
    i += 1
#fake_noise_3 = get_noise(1, z_dim, device='cuda')
#print(fake_noise_3.size())

#fake_3 = gen(fake_noise_3)
#save_image(fake_3, "fake_trained_" + str(i) + ".png")