# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Created by Shuangqin Cheng (sqcheng@stu2021.jnu.edu.cn)
# ------------------------------------------------------------------------------

import argparse
from lib.config.config import cfg_from_yaml, cfg, merge_dict_and_yaml, print_easy_dict
from lib.dataset.factory import get_dataset
from lib.model.factory import get_model
from lib.utils.visualizer import tensor_back_to_unnormalization, tensor_back_to_unMinMax
from lib.utils.metrics_np import MAE, MSE, Peak_Signal_to_Noise_Rate, Structural_Similarity, Cosine_Similarity,LPIPS,RMSE, NRMSE
import copy
import tqdm
import torch
import numpy as np
import os
from lib.model.lpips import PerceptualLoss
from scipy.stats import shapiro, ttest_1samp

def parse_args():
  parse = argparse.ArgumentParser(description='CTGAN')
  parse.add_argument('--data', type=str, default='', dest='data',
                     help='input data ')
  parse.add_argument('--tag', type=str, default='', dest='tag',
                     help='distinct from other try')
  parse.add_argument('--dataroot', type=str, default='', dest='dataroot',
                     help='input data root')
  parse.add_argument('--dataset', type=str, default='', dest='dataset',
                     help='Train or test or valid')
  parse.add_argument('--datasetfile', type=str, default='', dest='datasetfile',
                     help='Train or test or valid file path')
  parse.add_argument('--ymlpath', type=str, default=None, dest='ymlpath',
                     help='config have been modified')
  parse.add_argument('--gpu', type=str, default='0,1', dest='gpuid',
                     help='gpu is split by ,')
  parse.add_argument('--dataset_class', type=str, default='unalign', dest='dataset_class',
                     help='Dataset class should select from unalign /')
  parse.add_argument('--model_class', type=str, default='cyclegan', dest='model_class',
                     help='Model class should select from cyclegan / ')
  parse.add_argument('--check_point', type=str, default=None, dest='check_point',
                     help='which epoch to load? ')
  parse.add_argument('--latest', action='store_true', dest='latest',
                     help='set to latest to use latest cached model')
  parse.add_argument('--verbose', action='store_true', dest='verbose',
                     help='if specified, print more debugging information')
  parse.add_argument('--load_path', type=str, default=None, dest='load_path',
                     help='if load_path is not None, model will load from load_path')
  parse.add_argument('--how_many', type=int, dest='how_many', default=50,
                     help='if specified, only run this number of test samples for visualization')
  parse.add_argument('--resultdir', type=str, default='', dest='resultdir',
                     help='dir to save result')
  args = parse.parse_args()
  return args

def getShapiro(data):
   array = np.array(data, dtype = float)
   stat, p = shapiro(array)
   # 输出检验结果
   print('Shapiro-Wilk检验结果：统计量=%.3f, p值=%.3f' % (stat, p))
   
   # 根据p值进行正态性检验
   alpha = 0.05
   if p > alpha:
      print('数据符合正态分布')
   else:
      print('数据不符合正态分布')




def evaluate(args):
  # check gpu
  if args.gpuid == '':
    args.gpu_ids = []
  else:
    if torch.cuda.is_available():
      split_gpu = str(args.gpuid).split(',')
      args.gpu_ids = [int(i) for i in split_gpu]
    else:
      print('There is no gpu!')
      exit(0)

  # check point
  if args.check_point is None:
    args.epoch_count = 1
  else:
    args.epoch_count = int(args.check_point)

  # merge config with yaml
  if args.ymlpath is not None:
    cfg_from_yaml(args.ymlpath)
  # merge config with argparse
  opt = copy.deepcopy(cfg)
  opt = merge_dict_and_yaml(args.__dict__, opt)
  print_easy_dict(opt)

  opt.serial_batches = True

  # add data_augmentation
  datasetClass, _, dataTestClass, collateClass = get_dataset(opt.dataset_class)
  opt.data_augmentation = dataTestClass

  # get dataset
  dataset = datasetClass(opt)
  print('DataSet is {}'.format(dataset.name))
  dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=int(opt.nThreads),
    collate_fn=collateClass)

  dataset_size = len(dataloader)
  print('#Test images = %d' % dataset_size)

  # get model
  gan_model = get_model(opt.model_class)()
  print('Model --{}-- will be Used'.format(gan_model.name))

  # set to test
  gan_model.eval()

  gan_model.init_process(opt)
  total_steps, epoch_count = gan_model.setup(opt)

  # must set to test Mode again, due to  omission of assigning mode to network layers
  # model.training is test, but BN.training is training
  if opt.verbose:
    print('## Model Mode: {}'.format('Training' if gan_model.training else 'Testing'))
    for i, v in gan_model.named_modules():
      print(i, v.training)

  if 'batch' in opt.norm_G:
    gan_model.eval()
  elif 'instance' in opt.norm_G:
    gan_model.eval()
    # instance norm in training mode is better
    for name, m in gan_model.named_modules():
      if m.__class__.__name__.startswith('InstanceNorm'):
        m.train()
  else:
    raise NotImplementedError()

  if opt.verbose:
    print('## Change to Model Mode: {}'.format('Training' if gan_model.training else 'Testing'))
    for i, v in gan_model.named_modules():
      print(i, v.training)

  result_dir = os.path.join(opt.resultdir, opt.data, '%s_%s' % (opt.dataset, opt.check_point))
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)

  # ssim3d_loss = SSIM3D(window_size = 11)
  avg_dict = dict()
  percept = PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)
  for epoch_i, data in tqdm.tqdm(enumerate(dataloader)):

    gan_model.set_input(data)
    gan_model.test()

    visuals = gan_model.get_current_visuals()
    img_path = gan_model.get_image_paths()

    #
    # Evaluate Part
    #
    generate_CT = visuals['G_fake'].data.clone().cpu().numpy()
    real_CT = visuals['G_real'].data.clone().cpu().numpy()
    # To [0, 1]
    # To NDHW
    if 'std' in opt.dataset_class or 'baseline' in opt.dataset_class:
      generate_CT_transpose = generate_CT
      real_CT_transpose = real_CT
    else:
      generate_CT_transpose = np.transpose(generate_CT, (0, 2, 1, 3))
      real_CT_transpose = np.transpose(real_CT, (0, 2, 1, 3))
    generate_CT_transpose = tensor_back_to_unnormalization(generate_CT_transpose, opt.CT_MEAN_STD[0],
                                                           opt.CT_MEAN_STD[1])
    real_CT_transpose = tensor_back_to_unnormalization(real_CT_transpose, opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1])
    # clip generate_CT
    generate_CT_transpose = np.clip(generate_CT_transpose, 0, 1)

    # CT range 0-1
    mae0 = MAE(real_CT_transpose, generate_CT_transpose, size_average=False)
    mse0 = MSE(real_CT_transpose, generate_CT_transpose, size_average=False)
    cosinesimilarity = Cosine_Similarity(real_CT_transpose, generate_CT_transpose, size_average=False)
    ssim = Structural_Similarity(real_CT_transpose, generate_CT_transpose, size_average=False, PIXEL_MAX=1.0)
    lpips = LPIPS(torch.from_numpy(real_CT_transpose).unsqueeze(0), torch.from_numpy(generate_CT_transpose).unsqueeze(0),percept, size_average=False)
    # CT range 0-4096
    generate_CT_transpose = tensor_back_to_unMinMax(generate_CT_transpose, opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]).astype(
      np.int32)
    real_CT_transpose = tensor_back_to_unMinMax(real_CT_transpose, opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]).astype(
      np.int32)
    # psnr_3d = Peak_Signal_to_Noise_Rate_3D(real_CT_transpose, generate_CT_transpose, size_average=False, PIXEL_MAX=4095)
    psnr = Peak_Signal_to_Noise_Rate(real_CT_transpose, generate_CT_transpose, size_average=False, PIXEL_MAX=4095)
    mae = MAE(real_CT_transpose, generate_CT_transpose, size_average=False)
    mse = MSE(real_CT_transpose, generate_CT_transpose, size_average=False)
    rmse = RMSE(real_CT_transpose, generate_CT_transpose, size_average=False)
    nrmse = NRMSE(real_CT_transpose, generate_CT_transpose)

    name1 = os.path.splitext(os.path.basename(img_path[0][0]))[0]
    name2 = os.path.split(os.path.dirname(img_path[0][0]))[-1]
    name = name2 + '_' + name1

    ###SSIM_3D
    # ssim_3d = ssim3d_loss(torch.from_numpy(real_CT_transpose).unsqueeze(0), torch.from_numpy(generate_CT_transpose).unsqueeze(0)).numpy()
    print(cosinesimilarity, name)
    if cosinesimilarity is np.nan or cosinesimilarity > 1:
      print(os.path.splitext(os.path.basename(gan_model.get_image_paths()[0][0]))[0])
      continue

    metrics_list = [('MAE0', mae0), ('MSE0', mse0), ('MAE', mae), ('MSE', mse), ('RMSE', rmse),('NRMSE', nrmse),('CosineSimilarity', cosinesimilarity),
                    ('PSNR-TS', psnr[0]), ('SSIM-TS', ssim[0]), ('LPIPS-TS', lpips[0]),
                    ('PSNR-CS', psnr[1]), ('SSIM-CS', ssim[1]), ('LPIPS-CS', lpips[1]),
                    ('PSNR-SS', psnr[2]), ('SSIM-SS', ssim[2]),  ('LPIPS-SS', lpips[2]), 
                    ('PSNR-avg', psnr[3]),('SSIM-avg', ssim[3]), ('LPIPS-avg', lpips[3]),
                    ]
    

    for key, value in metrics_list:
      if avg_dict.get(key) is None:
        avg_dict[key] = [] + value.tolist()
      else:
        avg_dict[key].extend(value.tolist())

    del visuals, img_path
  
  # getShapiro(avg_dict['PSNR-TS'])

  psnr_last = []
  ssim_last = []
  lpips_last = []
  for key, value in avg_dict.items():
    print('### --{}-- total: {}; avg: {}; std: {} '.format(key, len(value), np.round(np.mean(value), 7),  np.round(np.std(value), 7)))
    if((key == 'PSNR-TS') or (key == 'PSNR-CS') or (key == 'PSNR-SS')): psnr_last.append(np.mean(value))
    if((key == 'SSIM-TS') or (key == 'SSIM-CS') or (key == 'SSIM-SS')): ssim_last.append(np.mean(value))
    if((key == 'LPIPS-TS') or (key == 'LPIPS-CS') or (key == 'LPIPS-SS')): lpips_last.append(np.mean(value))
    avg_dict[key] = np.mean(value)

  print('### --PSNR_std--  std: {} '.format( np.round(np.std(psnr_last), 7)))
  print('### --SSIM_std--  std: {} '.format( np.round(np.std(ssim_last), 7)))
  print('### --LPIPS_std--  std: {} '.format( np.round(np.std(lpips_last), 7)))
  return avg_dict


if __name__ == '__main__':
  args = parse_args()
  evaluate(args)
