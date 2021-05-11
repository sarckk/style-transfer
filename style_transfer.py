#!/usr/bin/env python
# coding: utf-8

import torch 
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st
import io

# 1_1 1_2      2_1. 2_2.      3_1. 3_2. 3_3. 3_4.      4_1. 4_2  4_3. 4_4.      5_1. 5_2. 5_3. 5_4.  
# 64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512, 'P'
normalize_stats = ([103.939, 116.779, 123.68], [1,1,1])
channels = [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512, 'P']

class GramMatrix(nn.Module):
  def forward(self, x):
    B, C, H, W = x.size() # here B must be 1 
    f = x.view(C, H * W) # row -> channel, columns -> individual activation laid out horizontally
    G = torch.mm(f, f.t())
    return G.div(f.nelement())

class VGG19(nn.Module):
  def __init__(self, content_layers, style_layers):
    super(VGG19, self).__init__()
    in_channels, block_number, layer_number = 3,1,1
    self.mode = 'capture'
    self.targets, self.losses = [{l_name: torch.empty(0) for l_name in content_layers+style_layers} for i in range(2)]
  
    layers = []
    for c in channels:
      if c == 'P':
        pool = nn.AvgPool2d(kernel_size=2, stride=2)
        layers.append(pool)
        block_number+=1
        layer_number=1
      else:
        layer_name = f"r{block_number}_{layer_number}"
        conv2d = nn.Conv2d(in_channels, c, kernel_size = 3, stride = 1, padding = 1)
        relu = nn.ReLU(inplace=True)
        if layer_name in content_layers:
          relu.register_forward_hook(self.capture_loss_hook(layer_name, True))
        elif layer_name in style_layers:
          relu.register_forward_hook(self.capture_loss_hook(layer_name, False))
        layers += [conv2d,relu]
        in_channels = c
        layer_number+=1 
    self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(output_size=(7,7)), nn.Flatten())
    self.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 1000),
    )
    
  def capture_loss_hook(self, layer_name, is_content_loss):
       def fn(module, _ ,output):
         if self.mode == 'content' and is_content_loss:
           self.targets[layer_name] = output.detach()
         elif self.mode == 'style' and not is_content_loss:
           self.targets[layer_name] = output.detach()
         elif self.mode == 'loss':
           target = self.targets[layer_name]
           if is_content_loss:
             self.losses[layer_name] = nn.MSELoss()(output, target)
           else:
             G = GramMatrix()
             self.losses[layer_name] = nn.MSELoss()(G(output), G(target))
       return fn

  def forward(self, x):
     out = self.features(x)
     out = self.classifier(out)
     return out

def preprocess(bytes, im_size):
  image = Image.open(io.BytesIO(bytes)) 
  image = image.convert('RGB')
  if type(im_size) is not tuple:
    im_size = tuple([int(im_size/max(image.size) * dim) for dim in (image.height, image.width)])
  tfms = transforms.Compose([
      transforms.Resize(im_size),
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x.mul_(255)),
      transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
      transforms.Normalize(*normalize_stats)
  ])
  return tfms(image)[None,:,:,:] # same as unsqueeze(0)

def get_model(content_layers, style_layers):
    vgg = VGG19(content_layers, style_layers)
    vgg19_pretrained = torch.hub.load('pytorch/vision:v0.9.0', 'vgg19', pretrained=True)
    vgg.load_state_dict(vgg19_pretrained.state_dict())
    return vgg

def setup_optim(img):
   optim = torch.optim.LBFGS([img])
   return optim


def deprocess(img):
  img = img.data[0].cpu()
  tfms = transforms.Compose([
    transforms.Normalize(mean=[-103.939, -116.779, -123.68], std=[1,1,1]),
    transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]),
  ])
  img = tfms(img) / 255
  img.clamp_(0,1)
  img = img.permute(1,2,0)
  return img.numpy()


def get_device():
  if torch.cuda.is_available():
    return torch.cuda.FloatTensor

  return torch.FloatTensor

def style_transfer(content_im, style_im, im_size, content_weight, style_weight, content_layers, style_layers, cb, num_steps):
  dtype = get_device()
  model = get_model(content_layers, style_layers)
  content_tens, style_tens = preprocess(content_im, im_size).type(dtype), preprocess(style_im, im_size).type(dtype)

  model.mode = 'content'
  model(content_tens) # capture losses

  model.mode = 'style'
  model(style_tens) # capture styles

  model.mode = 'loss'

  for param in model.parameters():
    param.requires_grad = False

  img = nn.Parameter(torch.randn(content_tens.shape).mul(0.001).type(dtype), requires_grad=True)
  optim = setup_optim(img)
  run = [0]

  def closure():
    run[0] += 1

    optim.zero_grad()

    model(img)

    style_loss = 0
    content_loss = 0

    for style_layer in style_layers:
      style_loss += model.losses[style_layer]
    for content_layer in content_layers:
      content_loss += model.losses[content_layer]

    style_loss *= style_weight
    content_loss *= content_weight
    loss = style_loss + content_loss
    loss.backward()

    if run[0] % 10 == 0:
      cb(run[0], num_steps, style_loss, content_loss)

    return loss


  while run[0] <= num_steps:
    optim.step(closure)
  
  return img 