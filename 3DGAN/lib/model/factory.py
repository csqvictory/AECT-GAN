# ------------------------------------------------------------------------------
# Copyright (c) Tencent

# Created by Shuangqin Cheng (sqcheng@stu2021.jnu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.optim import lr_scheduler


def get_model(dataset_name):
  if dataset_name == 'SingleViewCTGAN':
    from .singleView_CTGAN import CTGAN
    return CTGAN
  elif dataset_name == 'SingleView-SsACT-GANB':
    from .sin_SsACT_GANB import CTGAN
    return CTGAN
  elif dataset_name == 'MultiViewCTGAN':
    from .multiView_CTGAN import CTGAN
    return CTGAN
  elif dataset_name == 'MultiView-SsACT-GANB':
    from .mul_SsACT_GANB import CTGAN
    return CTGAN
  else:
    raise KeyError('Model class should select from simpleGan / GD2dGAN ')

def get_scheduler(optimizer, opt):
  if opt.lr_policy == 'lambda':
    def lambda_rule(epoch):
      lr_l = 1.0 - max(0, epoch + 1 + 0 - opt.niter) / float(opt.niter_decay + 1)
      return lr_l

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
  elif opt.lr_policy == 'step':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
  elif opt.lr_policy == 'plateau':
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
  elif opt.lr_policy == 'invariant':
    def lambda_rule(epoch):
      lr_l = 1.0
      return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
  else:
    return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
  return scheduler