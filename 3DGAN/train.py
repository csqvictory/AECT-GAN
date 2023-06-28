# ------------------------------------------------------------------------------
# Copyright (c) Tencent

# Created by Shuangqin Cheng (sqcheng@stu2021.jnu.edu.cn)
# ------------------------------------------------------------------------------

import argparse
from lib.config.config import cfg_from_yaml, cfg, merge_dict_and_yaml, print_easy_dict
from lib.dataset.factory import get_dataset
from lib.model.factory import get_model
import copy
import torch
import time
import os
import matplotlib.pyplot as plt
import numpy as np

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
  parse.add_argument('--valid_dataset', type=str, default=None, dest='valid_dataset',
                     help='Train or test or valid')
  parse.add_argument('--datasetfile', type=str, default='', dest='datasetfile',
                     help='Train or test or valid file path')
  parse.add_argument('--valid_datasetfile', type=str, default='', dest='valid_datasetfile',
                     help='Train or test or valid file path')
  parse.add_argument('--ymlpath', type=str, default=None, dest='ymlpath',
                     help='config have been modified')
  parse.add_argument('--gpu', type=str, default='0,1', dest='gpuid',
                     help='gpu is split by ,')
  parse.add_argument('--dataset_class', type=str, default='align', dest='dataset_class',
                     help='Dataset class should select from align /')
  parse.add_argument('--model_class', type=str, default='simpleGan', dest='model_class',
                     help='Model class should select from simpleGan / ')
  parse.add_argument('--check_point', type=str, default=None, dest='check_point',
                     help='which epoch to load? ')
  parse.add_argument('--load_path', type=str, default=None, dest='load_path',
                     help='if load_path is not None, model will load from load_path')
  parse.add_argument('--latest', action='store_true', dest='latest',
                     help='set to latest to use latest cached model')
  parse.add_argument('--verbose', action='store_true', dest='verbose',
                     help='if specified, print more debugging information')
  args = parse.parse_args()
  return args

if __name__ == '__main__':
  args = parse_args()

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
    args.epoch_count = int(args.check_point) + 1

  # merge config with yaml
  if args.ymlpath is not None:
    cfg_from_yaml(args.ymlpath)
  # merge config with argparse
  opt = copy.deepcopy(cfg)
  opt = merge_dict_and_yaml(args.__dict__, opt)
  print_easy_dict(opt)

  # add data_augmentation
  datasetClass, augmentationClass, dataTestClass, collateClass = get_dataset(opt.dataset_class)
  opt.data_augmentation = augmentationClass

  # valid dataset
  if args.valid_dataset is not None:
    valid_opt = copy.deepcopy(opt)
    valid_opt.data_augmentation = dataTestClass
    valid_opt.datasetfile = opt.valid_datasetfile


    valid_dataset = datasetClass(valid_opt)
    print('Valid DataSet is {}'.format(valid_dataset.name))
    valid_dataloader = torch.utils.data.DataLoader(
      valid_dataset,
      batch_size=1,
      shuffle=False,
      num_workers=int(6),
      collate_fn=collateClass)
    valid_dataset_size = len(valid_dataloader)
    print('#validation images = %d' % valid_dataset_size)
  else:
    valid_dataloader = None

  # get dataset
  dataset = datasetClass(opt)
  print('DataSet is {}'.format(dataset.name))
  dataloader = torch.utils.data.DataLoader(
    dataset,
    # batch_size=opt.batch_size,
    batch_size=2,
    shuffle=True,
    num_workers=int(4),
    pin_memory=True,
    collate_fn=collateClass)

  dataset_size = len(dataloader)
  print('#training images = %d' % dataset_size)

  # get model
  # 加载有保存的模型，并在其基础上继续训练
  # gan_model = torch.load('/root/autodl-nas/CTGan_AFFfuse/restult/CTGan_AFFfuse_2/d2_multiview2500/checkpoint/80/80_net_D.pth')
  gan_model = get_model(opt.model_class)()
  print('Model --{}-- will be Used'.format(gan_model.name))
  
  gan_model.init_process(opt)
  # gan_model.netD.load_state_dict(torch.load('/root/autodl-nas/CTGan_AFFfuse/restult/CTGan_AFFfuse_2/d2_multiview2500/checkpoint/80/80_net_D.pth'))
  # gan_model.netG.load_state_dict(torch.load('/root/autodl-nas/CTGan_AFFfuse/restult/CTGan_AFFfuse_2/d2_multiview2500/checkpoint/80/80_net_G.pth'))
  total_steps, epoch_count = gan_model.setup(opt)
  # total_steps, epoch_count = gan_model.load_networks(60,'/root/autodl-nas/CTGan_AFFfuse/restult/CTGan_AFFfuse_4/d2_multiview2500/checkpoint')
  # set to train
  # gan_model.train()

  # visualizer
  from lib.utils.visualizer import Visualizer
  visualizer = Visualizer(log_dir=os.path.join(gan_model.save_root, 'train_log'))

  total_steps = total_steps
  opt.epoch_count = epoch_count
  # train discriminator more
  dataloader_iter_for_discriminator = iter(dataloader)
  D_loss = []
  G_loss = []
  start_time = time.time()
  # train
  for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    iter_data_time = time.time()
    D_epoch_loss = 0
    D_epoch_loss_lpips = 0
    D_epoch_loss_mes = 0
    D_epoch_loss_d = 0
    G_epoch_loss = 0
    GGAN_epoch_loss = 0
    G_epoch_loss_lpips = 0
    idt_epoch_loss = 0
    map_epoch_loss = 0
    count = len(dataset)
    for epoch_i, data in enumerate(dataloader):
      data_len = data[0].size()[0]
      iter_start_time = time.time()

      total_steps += 1

      gan_model.set_input(data)
      t0 = time.time()
      gan_model.optimize_parameters()
      t1 = time.time()

      # if total_steps == 1:
      #   visualizer.add_graph(model=gan_model, input=gan_model.forward())

      # # visual gradient
      # if opt.verbose and total_steps % opt.print_freq == 0:
      #   for name, para in gan_model.named_parameters():
      #     visualizer.add_histogram('Grad_' + name, para.grad.data.clone().cpu().numpy(), step=total_steps)
      #     visualizer.add_histogram('Weight_' + name, para.data.clone().cpu().numpy(), step=total_steps)
      #   for name in gan_model.model_names:
      #     net = getattr(gan_model, 'net' + name)
      #     if hasattr(net, 'output_dict'):
      #       for name, out in net.output_dict.items():
      #         visualizer.add_histogram(name, out.numpy(), step=total_steps)

      # loss
      loss_dict = gan_model.get_current_losses()
      # visualizer.add_scalars('Train_Loss', loss_dict, step=total_steps)
      total_loss = visualizer.add_total_scalar('Total loss', loss_dict, step=total_steps)
      # visualizer.add_average_scalers('Epoch Loss', loss_dict, step=total_steps, write=False)
      # visualizer.add_average_scalar('Epoch total Loss', total_loss)

      # metrics
      # metrics_dict = gan_model.get_current_metrics()
      # visualizer.add_scalars('Train_Metrics', metrics_dict, step=total_steps)
      # visualizer.add_average_scalers('Epoch Metrics', metrics_dict, step=total_steps, write=False)

      

      # if total_steps % opt.print_img_freq == 0:
      #   visualizer.add_image('Image', gan_model.get_current_visuals(), gan_model.get_normalization_list(), total_steps)
      with torch.no_grad():
        temp_G_loss, temp_G_lpips_loss, temp_D_loss_total, temp_D_loss_d, temp_D_loss_lpips, temp_D_loss_mse= gan_model.return_D_G_epochLoss()
        D_epoch_loss += (temp_D_loss_total.item()*data_len)
        D_epoch_loss_d += (temp_D_loss_d.item()*data_len)
        D_epoch_loss_lpips += (temp_D_loss_lpips.item()*data_len)
        D_epoch_loss_mes += (temp_D_loss_mse.item()*data_len)
        
              
        GGAN_epoch_loss += loss_dict["G"]*data_len
        G_epoch_loss_lpips += (temp_G_lpips_loss.item()*data_len)
        # G_epoch_loss_lungseg += (temp_G_lungSeg_loss.item()*data_len)
        idt_epoch_loss += loss_dict["idt"]*data_len
        map_epoch_loss += loss_dict["map_m"]*data_len
        G_epoch_loss += (temp_G_loss.item()*data_len) 
              
      '''
      WGAN
      '''
      if (opt.critic_times - 1) > 0:
        for critic_i in range(opt.critic_times - 1):
          try:
            data = next(dataloader_iter_for_discriminator)
            gan_model.set_input(data)
            gan_model.optimize_D()
          except:
            dataloader_iter_for_discriminator = iter(dataloader)
      del(loss_dict)

    with torch.no_grad():        
        D_epoch_loss /= count
        D_epoch_loss_d /= count
        D_epoch_loss_lpips /= count
        D_epoch_loss_mes /= count
        G_epoch_loss_lpips /= count
        # G_epoch_loss_lungseg /= count
        G_epoch_loss /= count
        GGAN_epoch_loss /= count
        idt_epoch_loss /= count
        map_epoch_loss /= count
        D_loss.append(D_epoch_loss)
        G_loss.append(G_epoch_loss)
    # # save model every epoch
    # print('saving the latest model (epoch %d, total_steps %d)' %
    #       (epoch, total_steps))
    # gan_model.save_networks(epoch, total_steps, True)
    print('total step: {} timer: {:.4f} sec.'.format(total_steps, t1 - t0))
    print('epoch {}/{}, step{}:{} || total loss:{:.4f}'.format(epoch, opt.niter + opt.niter_decay,
                                                               epoch_i, dataset_size, D_epoch_loss+G_epoch_loss))
    print("every_epoch_time = ", time.time() - epoch_start_time )
    print('D_epoch_loss = %.5f' % D_epoch_loss)
    print('D_epoch_loss_lpips = %.5f' % D_epoch_loss_lpips)
    print('D_epoch_loss_mes = %.5f' %  D_epoch_loss_mes)
    print('D_epoch_loss_d = %.5f' % D_epoch_loss_d)
    print('G_epoch_loss = %.5f' % G_epoch_loss)
    print('GGAN_epoch_loss = %.5f' % GGAN_epoch_loss)
    print('idt_epoch_loss = %.5f' % idt_epoch_loss)
    print('map_epoch_loss = %.5f' % map_epoch_loss)
    print('G_epoch_loss_lpips = %.5f' % G_epoch_loss_lpips)
    # print('G_epoch_loss_lungseg = %.5f' % G_epoch_loss_lungseg)
    
    print('')
    # save model several epoch
    if epoch % opt.save_epoch_freq == 0 and epoch >= opt.begin_save_epoch:
      print('saving the model at the end of epoch %d, iters %d' %
            (epoch, total_steps))
      gan_model.save_networks(epoch, total_steps)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    ##########
    # For speed
    ##########
    # visualizer.add_image('Image_Epoch', gan_model.get_current_visuals(), gan_model.get_normalization_list(), epoch)
    # visualizer.add_average_scalers('Epoch Loss', None, step=epoch, write=True)
    # visualizer.add_average_scalar('Epoch total Loss', None, step=epoch, write=True)

    # visualizer.add_average_scalers('Epoch Metrics', None, step=epoch, write=True)

    # visualizer.add_scalar('Learning rate', gan_model.optimizers[0].param_groups[0]['lr'], epoch)
    lr = gan_model.update_learning_rate(epoch)
    if(lr < 0):
      print('epoch: ',epoch)
      break
    # # Test
    # if args.valid_dataset is not None:
    #   if epoch % opt.save_epoch_freq == 0 or epoch==1:
    #     gan_model.eval()
    #     iter_valid_dataloader = iter(valid_dataloader)
    #     for v_i in range(len(valid_dataloader)):
    #       data = next(iter_valid_dataloader)
    #       gan_model.set_input(data)
    #       gan_model.test()
    #
    #       if v_i < opt.howmany_in_train:
    #         visualizer.add_image('Test_Image', gan_model.get_current_visuals(), gan_model.get_normalization_list(), epoch*10+v_i, max_image=25)
    #
    #       # metrics
    #       metrics_dict = gan_model.get_current_metrics()
    #       visualizer.add_average_scalers('Epoch Test_Metrics', metrics_dict, step=total_steps, write=False)
    #     visualizer.add_average_scalers('Epoch Test_Metrics', None, step=epoch, write=True)
    #
    #     gan_model.train()
  end_time = time.time()
  print('训练100epoch所花费的时间是：%.5f h' %(end_time-start_time)/3600)
  plt.plot(range(1, len(D_loss)+1), D_loss, label='D_loss')
  plt.plot(range(1, len(D_loss)+1), G_loss, label='G_loss')
  plt.xlabel('epoch')
  plt.legend()
  plt.show()
  plt.savefig("/root/autodl-tmp/sobel_3dsslmse_lpips001_Glipips001/lossplt.png")
  np.savetxt('/root/autodl-tmp/sobel_3dsslmse_lpips001_Glipips001/D_Loss.txt',D_loss)
  np.savetxt('/root/autodl-tmp/sobel_3dsslmse_lpips001_Glipips001/G_Loss.txt',G_loss)
  