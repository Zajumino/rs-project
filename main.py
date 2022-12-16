#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from PIL import Image


# In[2]:


# sd of noise applied to image ~ N(0,sigma^2)
sigma = 0.1

# at each iteration we perturb the z with an additive noise ~ N(0,sigma_p^2)
sigma_p = 1./30

# learning rate
lr = 0.01

# To see overfitting set max_iter to a large value and remove stop condition
max_iter = 1800

# Number of channels in the output
nc = 3

# Size of z latent vector (generator input)
nz = 32

# no gpu :(
ngpu = 0


# In[3]:


device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# In[4]:


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return -1
    else:
        return 10 * np.log10(1. / mse)


# In[5]:


def intify(I):
    assert I.dtype == np.float32 or I.dtype == np.float64
    J = I * 255.0
    return J.astype(np.uint8)


def floatify(I):
    assert I.dtype == np.uint8
    J = I / 255.0
    return J.astype(np.float32)


# In[6]:


# weight init


# In[7]:


class CBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(CBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x):
        
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)

        return out


# In[8]:


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1, padding_mode='reflect')
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn(out)
        out = self.act(out)

        return out


# In[9]:


class UBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(UBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0)
        self.bn_in = nn.BatchNorm2d(in_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
    
    def forward(self, x):
        
        out = self.bn_in(x)
        out = self.conv1(out)
        out = self.bn(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn(out)
        out = self.act(out)
        out = self.up(out)

        return out


# In[10]:


class SBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    def forward(self, x):
        
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)

        return out


# In[11]:


class Hourglass(nn.Module):
    def __init__(self):
        super(Hourglass, self).__init__()
        self.ngpu = ngpu
        
        # down
        self.d1 = DBlock(nz, 128, 3)
        self.d2 = DBlock(128, 128, 3)
        self.d3 = DBlock(128, 128, 3)
        self.d4 = DBlock(128, 128, 3)
        self.d5 = DBlock(128, 128, 3)
        
        # up
        self.u5 = UBlock(132, 128, 3)
        self.u4 = UBlock(132, 128, 3)
        self.u3 = UBlock(132, 128, 3)
        self.u2 = UBlock(132, 128, 3)
        self.u1 = UBlock(132, 128, 3)
        
        # skip
        self.s1 = SBlock(128, 4, 1)
        self.s2 = SBlock(128, 4, 1)
        self.s3 = SBlock(128, 4, 1)
        self.s4 = SBlock(128, 4, 1)
        self.s5 = SBlock(128, 4, 1)
        
        self.conv = CBlock(128, nc, 1, 1)
        self.output = nn.Sigmoid()
        
    def forward(self, x):
        
        out = self.d1(x)
        skip1 = self.s4(out)
        out = self.d2(out)
        skip2 = self.s4(out)
        out = self.d3(out)
        skip3 = self.s4(out)
        out = self.d4(out)
        skip4 = self.s4(out)
        out = self.d5(out)
        skip5 = self.s5(out)
        
        out = self.u5(torch.cat([out,skip5],1))
        out = self.u4(torch.cat([out,skip4],1))
        out = self.u3(torch.cat([out,skip3],1))
        out = self.u2(torch.cat([out,skip2],1))
        out = self.u1(torch.cat([out,skip1],1))
        
        out = self.conv(out)
        out = self.output(out)
        
        return out


# In[12]:


# read image
img = plt.imread('lena.jpg')
if img.dtype == np.uint8:
    img = floatify(img).astype(np.float32) # convert to 32 bit float (0-1)

# generate noisy image
noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
noisy_img = np.clip(img + noise, 0, 1).astype(np.float32)
Image.fromarray((noisy_img*255).astype(np.uint8)).save('lena\lena_noisy.png')

# convert noisy image to tensor
x = torch.tensor(np.moveaxis(img, -1, 0), requires_grad=True).unsqueeze(0).to(device)
x0 = torch.tensor(np.moveaxis(noisy_img, -1, 0), requires_grad=True).unsqueeze(0).to(device)

# Notes
# .png is read as 0-1 float
# everything else is 0-255 int
# moveaxis changes the order of dimensions (so that it works as a tensor)
# unsqueeze adds a dimension -> (N=1, C=3, H=512, W=512)
# TENSOR.squeeze().detach().cpu().numpy() # convert back to numpy array
# np.moveaxis(numpy_array ,0,-1) # correct axis

# display images
plt.figure(0, figsize=(10,10))
plt.subplot(1, 3, 1); plt.imshow(img)
plt.subplot(1, 3, 2); plt.imshow((noise - noise.min()) / (noise.max() - noise.min()))
plt.subplot(1, 3, 3); plt.imshow(noisy_img)
plt.savefig('lena/lena_inputs.png')
plt.show()


# In[13]:


hg_net = Hourglass().to(device)
# print(hg_net)
# hg_net.apply(weights_init)

# number of parameters
s  = sum([np.prod(list(p.size())) for p in hg_net.parameters()]); 
print ('# of params: %d' % s)

# loss
mse = nn.MSELoss()

# net input
z = 0.1 * torch.rand(1, nz, 512, 512, device=device)


# In[14]:


optimizer = optim.Adam(hg_net.parameters(), lr=lr)

# Lists to keep track of progress
img_list = []
losses = []
psnr_t_mat = []
psnr_n_mat = []
iters = 0

for i in range(max_iter):
    
    #z = z + sigma_p * torch.randn(1, nz, 512, 512, device=device)
    out = hg_net(z) # run z through the network
    loss = mse(out,x0)
    
    psnr_noisy = psnr(x0.detach().cpu().numpy()[0], out.detach().cpu().numpy()[0])
    psnr_gt = psnr(x.detach().cpu().numpy()[0], out.detach().cpu().numpy()[0])
    
    optimizer.zero_grad() # reset the gradients of model parameters
    loss.backward() # backpropagate the prediction loss
    optimizer.step() # adjust the parameters by the gradients
    
    # Save Losses for plotting later
    losses.append(loss.item())
    psnr_t_mat.append(psnr_gt)
    psnr_n_mat.append(psnr_noisy)
    
    # Check how the network is doing
    if ((iters % 10 == 0) or (iters == max_iter-1)):
        with torch.no_grad():
            pred = out.detach().cpu()
            img_list.append(tv.utils.make_grid(pred, padding=2, normalize=True))
            print("iterations: [%d/%d]  loss: %f  psnr_gt: %f  psnr_noisy: %f" %(iters, max_iter, loss, psnr_gt, psnr_noisy))
            plt.imshow(np.moveaxis(img_list[-1].squeeze().detach().numpy(),0,-1))
            plt.show()
            if psnr_noisy >= 20:
                break
    
    iters += 1


# In[15]:


#torch.save(hg_net, 'filename.pth')
#hg_net = torch.load('filename.pth')



# In[16]:


img_out = np.moveaxis(img_list[-1].squeeze().detach().numpy(),0,-1)
plt.imsave('lena\lena_final.png', img_out)

for i in range(0, iters, 100):
    fname = 'lena\lena_iter%4.4d.png' %i
    arr = np.moveaxis(img_list[int(i/10)].squeeze().detach().numpy(),0,-1)
    plt.imsave(fname, arr)


# In[17]:


plt.plot(losses)
plt.xlabel('iteration')
plt.title('MSE')
plt.savefig('lena/lena_mse.png')
plt.show()


# In[18]:


plt.plot(psnr_t_mat, label='ground truth')
plt.plot(psnr_n_mat, label='noisy image')
plt.xlabel('iteration')
plt.legend(title='compared to')
plt.title('PSNR')
plt.savefig('lena/lena_psnr.png')
plt.show()


# In[ ]:




# In[ ]:



