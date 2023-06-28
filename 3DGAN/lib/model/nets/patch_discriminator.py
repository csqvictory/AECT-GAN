# ------------------------------------------------------------------------------
# Copyright (c) Tencent

# Created by Shuangqin Cheng (sqcheng@stu2021.jnu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch.nn as nn
import numpy as np
import functools
import torch
import torch.nn.functional as F
'''
Patch Discriminator
'''

class NLayer_2D_Discriminator(nn.Module):
  def __init__(self, input_nc, ndf=64, n_layers=3,
               norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False, n_out_channels=1):
    super(NLayer_2D_Discriminator, self).__init__()

    self.getIntermFeat = getIntermFeat
    self.n_layers = n_layers

    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
      use_bias = norm_layer == nn.InstanceNorm2d

    kw = 4
    padw = int(np.ceil((kw - 1.0) / 2))

    sequence = [[
      nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
      nn.LeakyReLU(0.2, True)
    ]]

    nf_mult = 1
    nf_mult_prev = 1
    for n in range(1, n_layers):
      nf_mult_prev = nf_mult
      nf_mult = min(2 ** n, 8)
      sequence += [[
        nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                  kernel_size=kw, stride=2, padding=padw, bias=use_bias),
        norm_layer(ndf * nf_mult),
        nn.LeakyReLU(0.2, True)]]

    nf_mult_prev = nf_mult
    nf_mult = min(2 ** n_layers, 8)
    sequence += [[
      nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                kernel_size=kw, stride=1, padding=padw, bias=use_bias),
      norm_layer(ndf * nf_mult),
      nn.LeakyReLU(0.2, True)
    ]]

    if use_sigmoid:
      sequence += [[nn.Conv2d(ndf * nf_mult, n_out_channels, kernel_size=kw, stride=1, padding=padw),
                    nn.Sigmoid()]]
    else:
      sequence += [[nn.Conv2d(ndf * nf_mult, n_out_channels, kernel_size=kw, stride=1, padding=padw)]]

    if getIntermFeat:
      for n in range(len(sequence)):
        setattr(self, 'model' + str(n), nn.Sequential(*(sequence[n])))
    else:
      sequence_stream = []
      for n in range(len(sequence)):
        sequence_stream += sequence[n]
      self.model = nn.Sequential(*sequence_stream)

  def forward(self, input):
    if self.getIntermFeat:
      res = [input]
      for n in range(self.n_layers + 2):
        model = getattr(self, 'model' + str(n))
        res.append(model(res[-1]))
      return res[1:]
    else:
      return [self.model(input)]


class DownBlockComp(nn.Module):
  def __init__(self, in_planes, out_planes):
    super(DownBlockComp, self).__init__()

    self.main = nn.Sequential(
        nn.Conv3d(in_planes, out_planes, 4, 2, 3, bias=False),
        nn.BatchNorm3d(out_planes), 
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv3d(out_planes, out_planes, 3, 1, 0, bias=False),
        nn.BatchNorm3d(out_planes), 
        nn.LeakyReLU(0.2)
        )

    self.direct = nn.Sequential(
        nn.AvgPool3d(2, 2),
        nn.Conv3d(in_planes, out_planes, 1, 1, 0, bias=False),
        nn.BatchNorm3d(out_planes), 
        nn.LeakyReLU(0.2))

  def forward(self, feat):
    return (self.main(feat) + self.direct(feat)) / 2

class Swish(nn.Module):
  def forward(self, feat):
      return feat * torch.sigmoid(feat)
        
class SEBlock(nn.Module):
  def __init__(self, ch_in, ch_out):
      super().__init__()

      self.main = nn.Sequential(
          nn.AdaptiveAvgPool3d(4), 
          nn.Conv3d(ch_in, ch_out, 4, 1, 0, bias=False), 
          Swish(),
          nn.Conv3d(ch_out, ch_out, 1, 1, 0, bias=False), 
          nn.Sigmoid())

  def forward(self, feat_small, feat_big):
      return feat_big * self.main(feat_small)

class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

class SimpleDecoder(nn.Module):
  """docstring for CAN_SimpleDecoder"""
  def __init__(self, nfc_in=64, nc=3):
    super(SimpleDecoder, self).__init__()

    nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
    nfc = {}
    for k, v in nfc_multi.items():
        nfc[k] = int(v*32)

    def upBlock(in_planes, out_planes):
      block = nn.Sequential(
          nn.Upsample(scale_factor=2, mode='nearest'),
          nn.Conv3d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
          nn.BatchNorm3d(out_planes * 2), 
          GLU()
      )
      return block

    self.main = nn.Sequential(
      nn.AdaptiveAvgPool3d(8),
      upBlock(nfc_in, nfc[16]) ,
      upBlock(nfc[16], nfc[32]),
      upBlock(nfc[32], nfc[64]),
      nn.Conv3d(nfc[64], nc, 3, 1, 1, bias=False),
      nn.Tanh()
    )
  def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)

class SimpleDecoder2(nn.Module):
  """docstring for CAN_SimpleDecoder"""
  def __init__(self, nfc_in=64, nc=3):
    super(SimpleDecoder2, self).__init__()

    nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
    nfc = {}
    for k, v in nfc_multi.items():
        nfc[k] = int(v*32)

    def upBlock(in_planes, out_planes):
      block = nn.Sequential(
          nn.Upsample(scale_factor=2, mode='nearest'),
          nn.Conv3d(in_planes, out_planes * 2, 3, 1, 1, bias=False),
          nn.BatchNorm3d(out_planes * 2), 
          GLU()
      )
      return block

    self.main = nn.Sequential(
      nn.AdaptiveAvgPool3d(16),
      upBlock(nfc_in, nfc[16]) ,
      upBlock(nfc[16], nfc[32]),
      nn.Conv3d(nfc[32], nc, 3, 1, 1, bias=False),
      nn.Tanh()
    )
  def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)

def crop_image_by_part(image, part):
    hw = image.size()[-1]//2
    if part==0:
        return image[:,:,:hw,:hw,:hw]
    if part==1:
        return image[:,:,:hw,:hw,hw:]
    if part==2:
        return image[:,:,:hw,hw:,:hw]
    if part==3:
        return image[:,:,:hw,hw:,hw:]
    if part==4:
        return image[:,:,hw:,:hw,:hw]
    if part==5:
        return image[:,:,hw:,:hw,hw:]
    if part==6:
        return image[:,:,hw:,hw:,:hw]
    if part==7:
        return image[:,:,hw:,hw:,hw:]

def crop_image_on_center(image):
  o = image.size()[-1]//2
  point1 = int(o - o//2)
  point2 = int(o + o//2)
  return image[:,:,point1:point2,point1:point2,point1:point2]
  


class NLayer_3D_Discriminator(nn.Module):
  def __init__(self, input_nc= 3, ndf=64, *args, **kwargs):
    super(NLayer_3D_Discriminator, self).__init__()
      
    self.down_from_big = nn.Sequential( 
        nn.Conv3d(input_nc, 64, 3, 1, 1, bias=False),
        nn.LeakyReLU(0.2, True)
    )

    self.down1  = DownBlockComp(64, 64)
    # self.se1 = SEBlock(64, 64)

    self.down2  = DownBlockComp(64, 128)
    self.se2 = SEBlock(64, 128)

    self.down3 = DownBlockComp(128, 256)
    self.se3 = SEBlock(128, 256)

    self.down4 = DownBlockComp(256, 512)
    self.se4 = SEBlock(256, 512)

    self.last = nn.Sequential( 
        nn.Conv3d(512, 64, 1, 1, 0, bias=False),
        nn.BatchNorm3d(64),
        nn.LeakyReLU(0.2, True),
        nn.Conv3d(64, 1, 4, 1, 0, bias=False)
    )

    self.simple_decoder_part = SimpleDecoder(nfc_in= 256, nc=3)
    self.simple_decoder_small = SimpleDecoder(nfc_in= 512, nc=3)
  

# 加一个label用于判断其输入的是否是Xreal

  def forward(self, inputs, label=True, part=None):
    # input.size() = torch.Size([1, 3, 128, 128, 128])
    x = self.down_from_big(inputs)  # torch.Size([4, 64, 128, 128, 128])

    x_2 = self.down1(x)             # torch.Size([4, 64, 64, 64, 64])

    x_4 = self.down2(x_2)
    x_4 = self.se2(x_2, x_4)        # torch.Size([4, 128, 32, 32, 32])

    x_8 = self.down3(x_4)
    x_8 = self.se3(x_4, x_8)       # torch.Size([4, 256, 16, 16, 16])

    x_16 = self.down4(x_8)
    x_16 = self.se4(x_8, x_16)       # torch.Size([4, 512, 8, 8, 8])

    

    out = self.last(x_16)           # torch.Size([4, 1, 5, 5, 5])

    if(label == True):
      assert part is not None
      rec_img_part = None
      rec_img_part = self.simple_decoder_part(crop_image_by_part(x_8, part))
      input_part = crop_image_by_part(inputs,part)
     
      x_16_d = self.simple_decoder_small(x_16)
      input_x_16_d = F.interpolate(inputs, size=[x_16_d.shape[2], x_16_d.shape[2],x_16_d.shape[2]])
      return [[rec_img_part,input_part], [x_16_d,input_x_16_d],out]
      
    
    return [out]



'''
Multi-Scale
Patch Discriminator
'''
#############################################################
# 3D Version
#############################################################
class Multiscale_3D_Discriminator(nn.Module):
  def __init__(self, input_nc, ndf=64, n_layers=3,
               norm_layer=nn.BatchNorm2d, use_sigmoid=False,
               getIntermFeat=False, num_D=3, n_out_channels=1):
    super(Multiscale_3D_Discriminator, self).__init__()
    assert num_D >= 1
    self.num_D = num_D
    self.n_layers = n_layers
    self.getIntermFeat = getIntermFeat

    for i in range(num_D):
      netD = NLayer_3D_Discriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat, n_out_channels)
      if getIntermFeat:
        for j in range(n_layers + 2):
          setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
      else:
        setattr(self, 'layer' + str(i), netD.model)

    self.downsample = nn.AvgPool3d(3, stride=2, padding=[1, 1, 1], count_include_pad=False)

  def singleD_forward(self, model, input):
    if self.getIntermFeat:
      result = [input]
      for i in range(len(model)):
        result.append(model[i](result[-1]))
      return result[1:]
    else:
      return [model(input)]

  def forward(self, input):
    num_D = self.num_D
    result = []
    input_downsampled = input
    for i in range(num_D):
      if self.getIntermFeat:
        model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in range(self.n_layers + 2)]
      else:
        model = getattr(self, 'layer' + str(num_D - 1 - i))

      result.append(self.singleD_forward(model, input_downsampled))
      if i != (num_D - 1):
        input_downsampled = self.downsample(input_downsampled)

    return result


#############################################################
# 2D Version
#############################################################
class Multiscale_2D_Discriminator(nn.Module):
  def __init__(self, input_nc, ndf=64, n_layers=3,
               norm_layer=nn.BatchNorm2d, use_sigmoid=False,
               getIntermFeat=False, num_D=3, n_out_channels=1):
    super(Multiscale_2D_Discriminator, self).__init__()
    self.num_D = num_D
    self.n_layers = n_layers
    self.getIntermFeat = getIntermFeat

    for i in range(num_D):
      netD = NLayer_2D_Discriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat, n_out_channels)
      if getIntermFeat:
        for j in range(n_layers + 2):
          setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
      else:
        setattr(self, 'layer' + str(i), netD.model)

    self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

  def singleD_forward(self, model, input):
    if self.getIntermFeat:
      result = [input]
      for i in range(len(model)):
        result.append(model[i](result[-1]))
      return result[1:]
    else:
      return [model(input)]

  def forward(self, input):
    num_D = self.num_D
    result = []
    input_downsampled = input
    for i in range(num_D):
      if self.getIntermFeat:
        model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
      else:
        model = getattr(self, 'layer' + str(num_D - 1 - i))

      result.append(self.singleD_forward(model, input_downsampled))
      if i != (num_D - 1):
        input_downsampled = self.downsample(input_downsampled)

    return result