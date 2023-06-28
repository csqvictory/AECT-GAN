# ------------------------------------------------------------------------------
# Copyright (c) Tencent

# Created by Shuangqin Cheng (sqcheng@stu2021.jnu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import

import torch

def Clear_blank_area(tensorA, blank_value_set=0):
  '''
  :param tensorA:
    CHW
  :param blank_value_set:
    default 0, given value of background
  :return:
  '''
  torch.sum(tensorA, )

def Offset_Register(tensorA, tensorB):
  '''
  :param tensorA:
    NCHW
  :param tensorB:
    NCHW
  :return:
  '''

