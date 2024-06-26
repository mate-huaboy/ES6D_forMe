#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings("ignore")
import random
import time
from copy import deepcopy
import shutil, time
import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import os
from torch.utils.tensorboard import SummaryWriter

from datasets.tless.tless_dataset import PoseDataset as pose_dataset

from models import ES6D as pose_net
import time
from lib.tless_evaluator import TLESSADDval
from lib.tless_gadd_evaluator import TLESSGADDval

from lib.utils import setup_logger
from lib.utils import warnup_lr, cal_mean_std
from lib.utils import post_processing_max as post_processing
# from lib.utils import post_processing_minloss as post_processing
# from lib.utils import post_processing_optimization as post_processing
import cv2
# import torchvision.transforms as transforms

from lib.utils import save_pred_and_gt_json
from lib.visual import vis_one_rotation_matrix as visualize_so3_distribution
st_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0, help='gpu number')

parser.add_argument('--experiment', type=str, default= "train",  help='brief description about experiment setting: train, test')

parser.add_argument('--loss_type', type=str, default= "Gaussian",  help='trianing loss: GADD, ADD, Gaussian')

parser.add_argument('--dataset', type=str, default='tless', help='ycb, tless')

parser.add_argument('--batch_size', type=int, default=64, help='batch size')

parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')

parser.add_argument('--lr', default=0.0001, help='learning rate, note that the learning rate at tless dataset is much larger than the ycb-video dataset')#0.002

parser.add_argument('--lr_rate', default=0.1, help='learning rate decay rate')

parser.add_argument('--warnup_iters', default=100, help='learning rate decay rate')#100

parser.add_argument('--decay_epoch', default=3, help='learning rate decay rate')#60

parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')

parser.add_argument('--augmentation', type=bool, default= False, help='train tless with data augmentation or not')

parser.add_argument('--nepoch', type=int, default=10, help='max number of epochs to train') #120

parser.add_argument('--resume', type=str, default='', help='resume ES6D model') #

parser.add_argument('--test_only', type=bool, default=False, help='resume es6d model') #

parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start') #

opt = parser.parse_args()


def main():

    # pre-setup
    global opt

    torch.backends.cudnn.enabled = True
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'#1235

    opt.gpu_number = torch.cuda.device_count()

    if opt.dataset == 'ycb':

        opt.num_objects = 21  # number of object classes in the dataset
        opt.num_points = 512
        opt.dataset_root = "./data/ycb"

        opt.outf = './experiments/ycb/{}/model'.format(opt.loss_type)  # folder to save trained models ########
        opt.log_dir = './experiments/ycb/{}/log'.format(opt.loss_type)  # folder to save logs ########

        if os.path.isdir(opt.outf) == False:
            os.makedirs(opt.outf)
        if os.path.isdir(opt.log_dir) == False:
            os.makedirs(opt.log_dir)


    elif opt.dataset == 'tless':
        opt.num_objects = 30  # number of object classes in the dataset
        opt.num_points = 1024
        opt.dataset_root = './datasets/tless'

        opt.outf = 'experiments/tless/{}/{}/model/'.format(opt.loss_type, opt.experiment)  # folder to save trained model
        opt.log_dir = 'experiments/tless/{}/{}/log/'.format(opt.loss_type, opt.experiment)  # folder to save logs ########

        if os.path.isdir(opt.outf) == False:
            os.makedirs(opt.outf)

        if os.path.isdir(opt.log_dir) == False:
            os.makedirs(opt.log_dir)

    mp.spawn(per_processor, nprocs=opt.gpu_number, args=(opt,))


#预测，调用训练和测试，测试时包括后处理过程以及可视化
def predict(data, estimator, lossor, opt, mode='train'):


    cls_ids = data['class_id'].to(opt.gpu)
    rgb = data['rgb'].to(opt.gpu)
    depth = data['xyz'].to(opt.gpu)
    mask = data['mask'].to(opt.gpu)
    gt_r = data['target_r'].to(opt.gpu)
    gt_t = data['target_t'].to(opt.gpu)

    # model_xyz = data['model_xyz'].cpu().numpy()
    model_xyz = data['model_xyz'].to(opt.gpu)

    #预测出分布的各个参数
    pre_t,pre_r,pre_s,pre_u = estimator(rgb, depth, cls_ids,mask)#测试也是知道cls_ids的

    if mode == 'train':
        loss, loss_dict = lossor(pre_t,pre_r,pre_s,pre_u, gt_r, gt_t,  cls_ids, model_xyz)
        return loss, loss_dict

    if mode == 'test':
        loss, loss_dict,pre_t,R_matrix,pre_s,pre_u = lossor(pre_t,pre_r,pre_s,pre_u, gt_r, gt_t,  cls_ids, model_xyz,is_train=False)
        print(data.keys())
        mean_xyz = data['mean_xyz'].cpu().numpy()#64*1*1*3

        #可视化

        # visualize_so3_distribution(R_matrix[0],pre_u[0],pre_s[0])
        visualize_so3_distribution(R_matrix[0],gt_r[0],pre_s[0])

        mean=[0.485*255.0, 0.456*255.0, 0.406*255.0]
        std=[0.229*255.0, 0.224*255.0, 0.225*255.0]
        cv2.imwrite("rgb.png",data['rgb'][0].numpy().transpose(1,2,0)*std+mean)
        
        # preds['xyz'] = depth

        # res_T = post_processing(preds, opt.sym_list)#b*3*4，由此得到最大的分数的pose，这里我们需要替换为最优算法
        #替换如下：
        #2）后处理：最优化or最大化
        res_R=post_processing(R_matrix,pre_s)#返回b*3*3
        # res_R=post_processing(R_matrix,pre_s,pre_u,5)#返回b*3*3
        # res_R=post_processing(R_matrix,pre_s,pre_u)#返回b*3*3

        pre_t=pre_t.unsqueeze(2)
        # res_T=torch.cat([res_R,gt_t.unsqueeze(2)],dim=2)
        res_T=torch.cat([res_R,pre_t],dim=2)
        bs, _, _ = res_T.size()

        res_T = res_T.cpu().numpy()

        tar_T = torch.cat([gt_r, gt_t.unsqueeze(dim=2)], dim=2)
        tar_T = tar_T.cpu().numpy()

        gt_cls = data['class_id'].cpu().numpy().astype(np.int)

        instance_id = data['instance_id'].cpu().numpy().astype(np.int)

        rt_list = []
        gt_rt_list = []
        gt_cls_list = []
        model_list = []

        instance_eval_rt_list =[]

        instance_id_list = []

        pred = res_T.copy()

        for i in range(bs):

            scale = opt.obj_radius[int(gt_cls[i])]

            instance_mean_xyz = mean_xyz[i][0,0,:]

            pred[i, :, 3] *= scale

            pred[i, :, 3] += instance_mean_xyz#预测出来的还得加上mean_xyz

            res_T[i, :, 3] *= scale#res_T没有加上mean_xyz
            tar_T[i, :, 3] *= scale
            model_xyz[i] *= scale

            instance_id_list.append([instance_id[i]])
            rt_list.append(res_T[i])#b*3*4
            gt_rt_list.append(tar_T[i])
            gt_cls_list.append(gt_cls[i] + 1)
            model_list.append(model_xyz[i].cpu().numpy())
            instance_eval_rt_list.append(pred[i])

        return loss, loss_dict, rt_list, gt_rt_list, gt_cls_list, model_list, instance_id_list, instance_eval_rt_list



def per_processor(gpu, opt):


    opt.gpu = gpu
    tensorboard_writer = 0
    if gpu == 0:
        tensorboard_writer = SummaryWriter(opt.log_dir)

    # init processor
    
    torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=gpu, world_size=opt.gpu_number)
    print("init gps:{}".format(gpu))
    torch.cuda.set_device(gpu)


    # init DDP model
    estimator = pose_net.ES6D(num_class=opt.num_objects,out_channel=32).to(gpu)
    estimator = torch.nn.parallel.DistributedDataParallel(estimator, device_ids=[gpu], output_device=gpu, find_unused_parameters=False)

    #选择要更新的网络参数
    # for name,p in estimator.named_parameters():
    #     if "full_net" in name or True:
    #         p.requires_grad=True
    #     else:
    #         p.requires_grad=False

    # init optimizer
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, estimator.parameters()), lr=opt.lr * opt.gpu_number)
    optimizer = optim.Adam(estimator.parameters(), lr=opt.lr * opt.gpu_number)

    
    # init DDP dataloader
    dataset = pose_dataset('train', opt.num_points, opt.dataset_root, True, opt.noise_trans)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False,
                                             num_workers=opt.workers, pin_memory=True, sampler=sampler)

    if gpu == 0:

        test_set = pose_dataset('test', opt.num_points, opt.dataset_root, False, opt.noise_trans)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size, shuffle=False,
                                                 num_workers=opt.workers*2, pin_memory=True)


    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()


    opt.obj_radius = dataset.obj_radius
    opt.raw_prim_groups = dataset.raw_prim_groups


    # init loss model
    train_loss = pose_net.get_loss(dataset = dataset, loss_type= opt.loss_type, train = True,out_channel=32).to(gpu)
    # test_loss = pose_net.get_loss(dataset=dataset, loss_type=opt.loss_type, train = False).to(gpu)

    # resume from existed model
    if opt.resume != '':
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load('{}'.format(opt.resume), map_location=loc)
            model_dict = estimator.state_dict()
            same_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict.keys()}
            model_dict.update(same_dict)
            estimator.load_state_dict(model_dict)
            # train_loss.load_state_dict(model_dict)
            # opt.start_epoch = checkpoint['epoch']
            opt.start_epoch = 0
            print("loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))

    # epoch loop
    tensorboard_loss_list = []
    tensorboard_test_list = []

    if opt.test_only:

        if gpu == 0:
            print('>>>>>>>>>>>test>>>>>>>>>>>')
            test(test_loader, estimator, train_loss, 0, tensorboard_writer, tensorboard_test_list, opt)
            torch.cuda.empty_cache()


    else:

        for epoch in range(opt.start_epoch, opt.nepoch + 1):


            sampler.set_epoch(epoch)
            opt.cur_epoch = epoch

            # # train for one epoch
            print('>>>>>>>>>>>train>>>>>>>>>>>')
            train(trainloader, estimator, train_loss, optimizer, epoch, tensorboard_writer, tensorboard_loss_list, opt)
            torch.cuda.empty_cache()

            # save checkpoint
            if gpu == 0 and epoch % 5 == 0 or epoch==opt.nepoch:
                print('>>>>>>>>>>>save checkpoint>>>>>>>>>>')
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': estimator.state_dict()},
                    '{}/checkpoint_{:04d}.pth.tar'.format(opt.outf, epoch))


            # test for one epoch
            if gpu == 0 and epoch % 5 == 0 or epoch==opt.nepoch:
                print('>>>>>>>>>>>test>>>>>>>>>>>')
                # train(test_loader, estimator, train_loss, optimizer, epoch, tensorboard_writer, tensorboard_loss_list, opt)
                test(test_loader, estimator, train_loss, epoch, tensorboard_writer, tensorboard_test_list, opt)
                torch.cuda.empty_cache()

        

        

def train(train_loader, estimator, lossor, optimizer, epoch, tensorboard_writer, tensorboard_loss_list, opt):

    if opt.gpu == 0:
        train_loss_list = []
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))

        if opt.gpu == 0:

            for key, value in sorted(vars(opt).items()):
                logger.info(str(key) + ': ' + str(value))

            # record
            logger.info('total train number : {}'.format(len(train_loader)) )


    estimator.train()
    optimizer.zero_grad()

    i = 0
    for data in train_loader:

        i += 1
        # update learning rate
        iter_th = epoch * len(train_loader) + i

        cur_lr = adjust_learning_rate(optimizer, epoch, iter_th, opt)

        loss, loss_dict = predict(data, estimator, lossor, opt, mode='train')


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # log and draw loss
        if opt.gpu == 0:

            train_loss_list.append(loss_dict)
            log_function(train_loss_list, logger, epoch, i, cur_lr)

            if len(train_loss_list) % 100 == 0:#之前是50
                l_dict = deepcopy(train_loss_list[-50])
                for ld in train_loss_list[-49:]:
                    for key in ld:
                        l_dict[key] += ld[key]
                for key in l_dict:
                    l_dict[key] = l_dict[key] / 50.0

                tensorboard_loss_list.append(l_dict)
                draw_loss_list('train', tensorboard_loss_list, tensorboard_writer)



def test(test_loader, estimator, lossor, epoch, tensorboard_writer, tensorboard_test_list, opt):

    if opt.gpu == 0:
        test_loss_list = []
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'test_%d_log.txt' % epoch))

        logger.info('total test number : {}'.format(len(test_loader)))

        # init evaluator
        tless_add_evaluator = TLESSADDval()
        tless_gadd_evaluator = TLESSGADDval()

    estimator.eval()

    raw_prim_groups_set = opt.raw_prim_groups

    with torch.no_grad():

        i = 0
        total_rt_list = []
        total_gt_list = []
        total_cls_list = []
        total_instance_list = []
        total_RT_list = []

        for data in test_loader:

            i += 1

            
            start_time = time.time()
            _, test_loss_dict, rt_list, gt_rt_list, gt_cls_list, model_list, instance_id_list, instance_eval_rt_list  = predict(data, estimator, lossor, opt, mode='test')
            end_time = time.time()
            print("运行时间：{}s".format(end_time-start_time))
            
            total_rt_list += rt_list
            total_gt_list += gt_rt_list
            total_cls_list += gt_cls_list
            total_instance_list += instance_id_list
            total_RT_list += instance_eval_rt_list

            # eval
            tless_add_evaluator.eval_pose_parallel(rt_list, gt_cls_list, gt_rt_list, gt_cls_list, model_list)#计算好像只是使用了没有加mean——xyz的rt——list，而不是instance

            model_list = []
            for gt_cls in gt_cls_list:
                model_list.append(raw_prim_groups_set[gt_cls[0]-1])

            tless_gadd_evaluator.eval_pose_parallel(rt_list, gt_cls_list, gt_rt_list, gt_cls_list, model_list)


            # log and draw loss
            if opt.gpu == 0:
                # log
                test_loss_list.append(test_loss_dict)
                log_function(test_loss_list, logger, epoch, i, opt.lr)

        save_pred_and_gt_json(total_RT_list, total_instance_list, total_gt_list, total_cls_list, opt.log_dir)#这里保存的又是total_RT_list，但是好像评估并没有用上

        # draw loss
        if opt.gpu == 0:

            # evaluation result
            add_cur_eval_info_dict = tless_add_evaluator.cal_auc()
            gadd_cur_eval_info_dict = tless_gadd_evaluator.cal_auc()

            # draw
            l = deepcopy(test_loss_list[0])

            for ld in test_loss_list[1:]:
                for key in ld:
                    l[key] += ld[key]
            for key in l:
                l[key] = l[key] / len(test_loss_list)

            l['add_auc'] = add_cur_eval_info_dict['auc']
            l['gadd_auc'] = gadd_cur_eval_info_dict['auc']

            tensorboard_test_list.append(l)
            draw_loss_list('test', tensorboard_test_list, tensorboard_writer)

            # output test result
            log_tmp = 'TEST ENDING: '

            for key in l:

                log_tmp = log_tmp + ' {}:{:.4f}'.format(key, l[key])

            logger.info(log_tmp)



def adjust_learning_rate(optimizer, epoch, iter, opt):

    """Decay the learning rate based on schedule"""
    lr = opt.lr * opt.gpu_number

    lr *= opt.lr_rate if epoch >= opt.decay_epoch else 1.

    if (iter <= opt.warnup_iters):
        lr = warnup_lr(iter, opt.warnup_iters, lr / 10, lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def log_function(loss_list, logger, epoch, batch, lr):
    l = loss_list[-1]
    tmp = 'time{} E{} B{} lr:{:.9f}'.format(
        time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, batch, lr)
    for key in l:
        tmp = tmp + ' {}:{:.4f}'.format(key, l[key])
    logger.info(tmp)

def draw_loss_list(phase, loss_list, tensorboard_writer):

    loss = loss_list[-1]

    for key in loss:

        tensorboard_writer.add_scalar(phase+'/'+key, loss[key], len(loss_list))


if __name__ == '__main__':

    main()

    # envs mypose

