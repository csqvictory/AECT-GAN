# ------------------------------------------------------------------------------
# Copyright (c) Tencent

# Created by Shuangqin Cheng (sqcheng@stu2021.jnu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import numpy as np
from lib.model.base_model import Base_Model
import lib.model.nets.factory as factory
from .loss.multi_gan_loss import GANLoss, RestructionLoss
from lib.utils.image_pool import ImagePool
import lib.utils.metrics as Metrics
import random
import torch.nn as nn
import torch.nn.functional as F
from lib.model.lpips import PerceptualLoss
percept = PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)


class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]
        kernel_h = [[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def get_gray(self,x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x
    

def get_cut_3dt2d(data):
  batch_size = data.shape[0]
  channel = data.shape[1]
  depth = data.shape[2]
  height = data.shape[3]
  width = data.shape[4]
  x = data.transpose(1,2)
  x = x.reshape(batch_size*depth, channel, height,width)
  return x

def get_discirmator_realimage_loss(data1,data2):
  crop_part1 = get_cut_3dt2d(data1[0])
  crop_part2 = get_cut_3dt2d(data1[1])
  image_small1 = get_cut_3dt2d(data2[0])
  image_small2 = get_cut_3dt2d(data2[1])
  lossans = percept(crop_part1,crop_part2).sum() + percept(image_small1,image_small2).sum()
  return lossans

def get_generator_featuremap_loss(data1,data2):
  fm1 = get_cut_3dt2d(data1)
  fm2 = get_cut_3dt2d(data2)
  lossans = percept(fm1,fm2).sum()
  return lossans

class CTGAN(Base_Model):
  def __init__(self):
    super(CTGAN, self).__init__()

  @property
  def name(self):
    return 'singleView_CTGAN'

  '''
  Init network architecture
  '''
  def init_network(self, opt):
    Base_Model.init_network(self, opt)
    self.gradLayer = GradLayer()
    self.if_pool = opt.if_pool
    self.multi_view = opt.multi_view
    self.conditional_D = opt.conditional_D
    assert len(self.multi_view) > 0

    self.loss_names = ['D', 'G']
    self.metrics_names = ['Mse', 'CosineSimilarity', 'PSNR']
    self.visual_names = ['G_real', 'G_fake', 'G_input', 'G_Map_fake_F', 'G_Map_real_F', 'G_Map_fake_S', 'G_Map_real_S']

    # identity loss
    if self.opt.idt_lambda > 0:
      self.loss_names += ['idt']

    # feature metric loss
    if self.opt.feature_D_lambda > 0:
      self.loss_names += ['fea_m']

    # map loss
    if self.opt.map_projection_lambda > 0:
      self.loss_names += ['map_m']

    if self.training:
      self.model_names = ['G', 'D']
    else:  # during test time, only load Gs
      self.model_names = ['G']

    self.netG = factory.define_3DG(opt.noise_len, opt.input_shape, opt.output_shape,
                                   opt.input_nc_G, opt.output_nc_G, opt.ngf, opt.which_model_netG,
                                   opt.n_downsampling, opt.norm_G, not opt.no_dropout,
                                   opt.init_type, self.gpu_ids, opt.n_blocks,
                                   opt.encoder_input_shape, opt.encoder_input_nc, opt.encoder_norm,
                                   opt.encoder_blocks, opt.skip_number, opt.activation_type, opt=opt)

    if self.training:
      # out of discriminator is not probability when
      # GAN loss is LSGAN
      use_sigmoid = False

      # conditional Discriminator
      if self.conditional_D:
        d_input_channels = opt.input_nc_D + 1
      else:
        d_input_channels = opt.input_nc_D
      self.netD = factory.define_D(d_input_channels, opt.ndf,
                                   opt.which_model_netD,
                                   opt.n_layers_D, opt.norm_D,
                                   use_sigmoid, opt.init_type, self.gpu_ids,
                                   opt.discriminator_feature, num_D=opt.num_D, n_out_channels=opt.n_out_ChannelsD)
      if self.if_pool:
        self.fake_pool = ImagePool(opt.pool_size)

  # correspond to visual_names
  def get_normalization_list(self):
    return [
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.XRAY1_MEAN_STD[0], self.opt.XRAY1_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
      [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]]
    ]

  def init_loss(self, opt):
    Base_Model.init_loss(self, opt)

    # #####################
    # define loss functions
    # #####################
    # GAN loss
    self.criterionGAN = GANLoss(use_lsgan=True).to(self.device)

    # identity loss
    self.criterionIdt = RestructionLoss(opt.idt_loss, opt.idt_reduction).to(self.device)

    # feature metric loss
    self.criterionFea = torch.nn.L1Loss().to(self.device)

    # map loss
    self.criterionMap = RestructionLoss(opt.map_projection_loss).to(self.device)

    # #####################
    # initialize optimizers
    # #####################
    self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                        lr=opt.lr, betas=(opt.beta1, opt.beta2))
    self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                        lr=opt.lr, betas=(opt.beta1, opt.beta2))
    self.optimizers = []
    self.optimizers.append(self.optimizer_G)
    self.optimizers.append(self.optimizer_D)

  '''
    Train -Forward and Backward
  '''
  def set_input(self, input):
    self.input_xray1_sober = self.gradLayer(input[1])
    self.G_input = input[1].to(self.device)
    self.G_real = input[0].to(self.device)
    self.image_paths = input[2:]

  # map function
  def output_map(self, v, dim):
    '''
    :param v: tensor
    :param dim:  dimension be reduced
    :return:
      N1HW
    '''
    ori_dim = v.dim()
    # tensor [NDHW]
    if ori_dim == 4:
      map = torch.mean(torch.abs(v), dim=dim)
      # [NHW] => [NCHW]
      return map.unsqueeze(1)
    # tensor [NCDHW] and c==1
    elif ori_dim == 5:
      # [NCHW]
      map = torch.mean(torch.abs(v), dim=dim)
      return map
    else:
      raise NotImplementedError()

  def transition(self, predict):
    p_max, p_min = predict.max(), predict.min()
    new_predict = (predict - p_min) / (p_max - p_min)
    return new_predict

  def ct_unGaussian(self, value):
    return value * self.opt.CT_MEAN_STD[1] + self.opt.CT_MEAN_STD[0]

  def ct_Gaussian(self, value):
    return (value - self.opt.CT_MEAN_STD[0]) / self.opt.CT_MEAN_STD[1]

  def post_process(self, attributes_name):
    if not self.training:
      if self.opt.CT_MEAN_STD[0] == 0 and self.opt.CT_MEAN_STD[0] == 0:
        for name in attributes_name:
          setattr(self, name, torch.clamp(getattr(self, name), 0, 1))
      elif self.opt.CT_MEAN_STD[0] == 0.5 and self.opt.CT_MEAN_STD[0] == 0.5:
        for name in attributes_name:
          setattr(self, name, torch.clamp(getattr(self, name), -1, 1))
      else:
        raise NotImplementedError()

  def projection_visual(self):
    # map F is projected in dimension of H
    self.G_Map_real_F = self.transition(self.output_map(self.ct_unGaussian(self.G_real), 2))
    self.G_Map_fake_F = self.transition(self.output_map(self.ct_unGaussian(self.G_fake), 2))
    # map S is projected in dimension of W
    self.G_Map_real_S = self.transition(self.output_map(self.ct_unGaussian(self.G_real), 3))
    self.G_Map_fake_S = self.transition(self.output_map(self.ct_unGaussian(self.G_fake), 3))

  def metrics_evaluation(self):
    # 3D metrics including mse, cs and psnr
    g_fake_unNorm = self.ct_unGaussian(self.G_fake)
    g_real_unNorm = self.ct_unGaussian(self.G_real)

    self.metrics_Mse = Metrics.Mean_Squared_Error(g_fake_unNorm, g_real_unNorm)
    self.metrics_CosineSimilarity = Metrics.Cosine_Similarity(g_fake_unNorm, g_real_unNorm)
    self.metrics_PSNR = Metrics.Peak_Signal_to_Noise_Rate(g_fake_unNorm, g_real_unNorm, PIXEL_MAX=1.0)

  def dimension_order_std(self, value, order):
    # standard CT dimension
    return value.permute(*tuple(np.argsort(order)))

  def forward(self):
    '''
    self.G_fake is generated object
    self.G_real is GT object
    '''
    # G_fake_D is [B 1 D H W]
    self.G_fake_D1 = self.netG([self.G_input,self.input_xray1_sober])
    self.G_fake_D = self.dimension_order_std(self.G_fake_D1, self.opt.CTOrder_Xray1)
    # visual object should be [B D H W]
    self.G_fake = torch.squeeze(self.G_fake_D, 1)
    # input of Discriminator is [B 1 D H W]
    self.G_real_D = torch.unsqueeze(self.G_real, 1)
    # if add condition to discriminator, expanding x-ray
    # as the same shape and dimension order as CT
    if self.conditional_D and self.training:
      self.G_condition_D = self.dimension_order_std(self.G_input.unsqueeze(1).expand_as(self.G_real_D), self.opt.CTOrder_Xray1).detach()
    # post processing, used only in testing
    self.post_process(['G_fake'])
    if not self.training:
      # visualization of x-ray projection
      self.projection_visual()
      # metrics
      self.metrics_evaluation()
    # multi-view projection maps for training
    # Note: self.G_real_D and self.G_fake_D are in dimension order of 'NCDHW'
    if self.training:
      for i in self.multi_view:
        out_map = self.output_map(self.ct_unGaussian(self.G_real_D), i + 1)
        out_map = self.ct_Gaussian(out_map)
        setattr(self, 'G_Map_{}_real'.format(i), out_map)

        out_map = self.output_map(self.ct_unGaussian(self.G_fake_D), i + 1)
        out_map = self.ct_Gaussian(out_map)
        setattr(self, 'G_Map_{}_fake'.format(i), out_map)

  # feature metrics loss
  def feature_metric_loss(self, D_fake_pred, D_real_pred, loss_weight, num_D, feat_weights, criterionFea):
    fea_m_lambda = loss_weight
    loss_G_fea = 0
    feat_weights = feat_weights
    D_weights = 1.0 / num_D
    weight = feat_weights * D_weights

    # multi-scale discriminator
    if isinstance(D_fake_pred[0], list):
      for di in range(num_D):
        for i in range(len(D_fake_pred[di]) - 1):
          loss_G_fea += weight * criterionFea(D_fake_pred[di][i], D_real_pred[di][i].detach()) * fea_m_lambda
    # single discriminator
    else:
      for i in range(len(D_fake_pred) - 1):
        loss_G_fea += feat_weights * criterionFea(D_fake_pred[i], D_real_pred[i].detach()) * fea_m_lambda

    return loss_G_fea
  
  def getVolumeLoss(self,data1,data2,idt_lambda):
    crop_Volume_loss = self.criterionIdt(data1[0], data1[1]) * idt_lambda
    resize_Volume_loss = self.criterionIdt(data2[0], data2[1]) * idt_lambda
    D_Volume_loss = crop_Volume_loss + resize_Volume_loss
    return D_Volume_loss   


  def get_cut_3dt2d(self, data):
    batch_size = data.shape[0]
    channel = data.shape[1]
    depth = data.shape[2]
    height = data.shape[3]
    width = data.shape[4]
    x = data.transpose(1,2)
    x = x.reshape(batch_size*depth, channel, height,width)
    x = torch.cat((x, torch.zeros_like(x[:, :1, :, :])), dim=1)
    return x

  def get_discirmator_realimage_loss(self,data1,data2):
    crop_part1 = self.get_cut_3dt2d(data1[0])
    crop_part2 = self.get_cut_3dt2d(data1[1])
    image_small1 = self.get_cut_3dt2d(data2[0])
    image_small2 = self.get_cut_3dt2d(data2[1])
    lossans = percept(crop_part1,crop_part2).sum() + percept(image_small1,image_small2).sum()
    return lossans
  
  def backward_D_basic(self, D_network, input_real, input_fake, fake_pool, criterionGAN, loss_weight):
  
    part = random.randint(0, 7)
    D_real_pred = D_network(input_real, True, part)
    gan_loss_real = criterionGAN(D_real_pred, True)

    if self.if_pool:
      g_fake_pool = fake_pool.query(input_fake)
    else:
      g_fake_pool = input_fake
    D_fake_pool_pred = D_network(g_fake_pool.detach(),False)
    gan_loss_fake = criterionGAN(D_fake_pool_pred, False)

    gan_D_volumeMse = self.getVolumeLoss(D_real_pred[0], D_real_pred[1],self.opt.idt_lambda)
    gan_D_plips_loss = self.get_discirmator_realimage_loss(D_real_pred[0], D_real_pred[1])*self.opt.dPecpt_D_lambda
    gan_loss = (gan_loss_fake + gan_loss_real) * loss_weight + gan_D_volumeMse + gan_D_plips_loss
    gan_loss.backward()
    return gan_loss, gan_loss-gan_D_volumeMse-gan_D_plips_loss, gan_D_plips_loss, gan_D_volumeMse

  def backward_D(self):
    if self.conditional_D:
      fake_input = torch.cat([self.G_condition_D, self.G_fake_D], 1)
      real_input = torch.cat([self.G_condition_D, self.G_real_D], 1)
    else:
      fake_input = self.G_fake_D
      real_input = self.G_real_D
    self.loss_D, self.D_loss, self.gan_D_plips_loss,self.gan_D_volume_mse = self.backward_D_basic(self.netD, real_input, fake_input,self.fake_pool if self.if_pool else None, self.criterionGAN, self.opt.gan_lambda)

  def backward_G_basic(self, D_network, input_fake, criterionGAN, loss_weight):
    D_fake_pred = D_network(input_fake, label=False, part=None)
    loss_G = criterionGAN(D_fake_pred, True) * loss_weight
    return loss_G, D_fake_pred

  def backward_G(self):
    idt_lambda = self.opt.idt_lambda
    fea_m_lambda = self.opt.feature_D_lambda
    map_m_lambda = self.opt.map_projection_lambda

    ############################################
    # BackBone GAN
    ############################################
    # GAN loss
    if self.conditional_D:
      fake_input = torch.cat([self.G_condition_D, self.G_fake_D], 1)
      real_input = torch.cat([self.G_condition_D, self.G_real_D], 1)
    else:
      fake_input = self.G_fake_D
      real_input = self.G_real_D

    self.loss_G, D_fake_pred = self.backward_G_basic(self.netD, fake_input, self.criterionGAN, self.opt.gan_lambda)

    # identity loss
    if idt_lambda > 0:
      # focus area weight assignment
      if self.opt.idt_reduction == 'none' and self.opt.idt_weight > 0:
        idt_low, idt_high = self.opt.idt_weight_range
        idt_weight = self.opt.idt_weight
        loss_idt = self.criterionIdt(self.G_fake_D, self.G_real_D)
        mask = (self.G_real_D > idt_low) & (self.G_real_D < idt_high)
        loss_idt[mask] = loss_idt[mask] * idt_weight
        self.loss_idt = loss_idt.mean() * idt_lambda
      else:
        self.loss_idt = self.criterionIdt(self.G_fake_D, self.G_real_D) * idt_lambda
        self.G_volume_loss = get_generator_featuremap_loss(self.G_fake_D, self.G_real_D) * self.opt.gPecpt_G_lambda
    

    # map loss
    if map_m_lambda > 0:
      self.loss_map_m = 0.
      for direction in self.multi_view:
        self.loss_map_m += self.criterionMap(
          getattr(self, 'G_Map_{}_fake'.format(direction)),
          getattr(self, 'G_Map_{}_real'.format(direction))) * map_m_lambda
      self.loss_map_m = self.loss_map_m / len(self.multi_view)

    # 0.0 must be add to loss_total, otherwise loss_G will
    # be changed when loss_total_G change
    self.loss_total_G = self.loss_G + 0.0 + self.G_volume_loss
    if idt_lambda > 0:
      self.loss_total_G += self.loss_idt
    if fea_m_lambda > 0:
      self.loss_total_G += self.loss_fea_m
    if map_m_lambda > 0:
      self.loss_total_G += self.loss_map_m

    self.loss_total_G.backward()

  def optimize_parameters(self):
    # forward
    self()
    self.set_requires_grad([self.netD], False)
    self.optimizer_G.zero_grad()
    self.backward_G()
    self.optimizer_G.step()

    self.set_requires_grad([self.netD], True)
    self.optimizer_D.zero_grad()
    self.backward_D()
    self.optimizer_D.step()

  def optimize_D(self):
    # forward
    self()
    self.set_requires_grad([self.netD], True)
    self.optimizer_D.zero_grad()
    self.backward_D()
    self.optimizer_D.step()

  def return_D_G_epochLoss(self):
    return self.loss_total_G, self.G_volume_loss, self.loss_D, self.D_loss, self.gan_D_plips_loss,self.gan_D_volume_mse
  


