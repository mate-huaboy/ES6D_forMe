#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import md_resnet18 as resnet_extractor
from models.pointnet import md_pointnet as spatial_encoder
from models.pointnet_util import knn_one_point
import numpy as np
from lib.utils import pose_from_predictions_train,ortho6d_to_mat_batch,batch_rotation_matrix_to_quaternion,quaternion_to_rotation_matrix
from models.loss import loss_by_bind



def get_header(in_channel, out_channel):
    return nn.Sequential(
            nn.Conv2d(in_channel, 640, kernel_size=1),
            nn.BatchNorm2d(640),
            nn.ReLU(inplace=True),
            nn.Conv2d(640, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channel, kernel_size=1)
        )

def get_full(in_channel,out_channel):#添加全连接层
    return nn.Sequential(
        nn.Linear(in_channel,1024),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Linear(1024,256),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Linear(256,out_channel)
    )


class XYZNet(nn.Module):
    def __init__(self,
                 in_channel=3,
                 strides=[2, 2, 1],
                 pn_conv_channels=[128, 128, 256, 512]):
        super(XYZNet, self).__init__()
        self.ft_1 = resnet_extractor(in_channel, strides)
        self.ft_2 = spatial_encoder(1024, pn_conv_channels)

    def forward(self, xyzrgb):
        ft_1 = self.ft_1(xyzrgb)
        b, c, h, w = ft_1.size()
        rs_xyz = F.interpolate(xyzrgb[:, :3], (h, w), mode='nearest')
        ft_2 = self.ft_2(ft_1, rs_xyz)
        ft_3 = torch.cat([ft_1, ft_2], dim=1)
        return ft_3, rs_xyz
    
#模型
class ES6D(nn.Module):
    def __init__(self, num_class=21,out_channel=32):
        super(ES6D, self).__init__()
        self.channel=out_channel
        self.num_class = num_class

        self.xyznet = XYZNet(6)

        self.trans = get_header(1024 + 512 + 512, 3 * num_class)

        self.prim_x = get_header(1024 + 512 + 512, 4 * num_class)

        self.score = get_header(1024 + 512 + 512, num_class)
        
        self.full_net=get_full(9*32*32,3+out_channel*9)#输出通道，同时估计sigmal和旋转，同时也要估计平移，
        # self.full_net=get_full(9*32*32,3+out_channel*6)#输出通道，使用四元数的方式

    def add_full(self, preds, mask):
        pred_r = preds['pred_r']
        pred_t = preds['pred_t']
        pred_score = preds['pred_s']

        bs, c, h, w = pred_r.size()
        pred_r = pred_r.view(bs, 4, h, w)
        pred_r = pred_r / torch.norm(pred_r, dim=1, keepdim=True)#应该是归一化
        pred_r = pred_r.view(bs, 4, -1)
        pred_t = pred_t.view(bs, 3, -1)
        pred_score = pred_score.view(bs, -1)
        mask = F.interpolate(mask, size=(h, w)).squeeze(dim=1)
        pred_score=pred_score.view(bs,1,-1)
        mask=mask.view(bs,1,-1)
        In=torch.cat([pred_r,pred_t,pred_score,mask],dim=1)
        In=In.view(bs,-1)
        out=self.full_net(In)#
        #out的
        pre_t=out[:,:3]
        pre_r=out[:,3:3+6*self.channel]#使用R_6d参数化方式
        pre_r=pre_r.view(bs,-1,6)
        pre_s=out[:,3+6*self.channel:3+7*self.channel]
        pre_s=F.softmax(nn.Tanh()(pre_s))
        # pre_s=F.softmax(nn.LeakyReLU(0.1, inplace=True)(pre_s))
        # pre_u=out[:,3+5*self.channel:].view(bs,-1,2)#64*32*2
        pre_u=out[:,3+7*self.channel:]#64*32*2

        #pre_u是有大小限制的
        min_kappa = 0.01
        pre_u=F.elu(pre_u)+1.0+min_kappa
        return pre_t,pre_r,pre_s,pre_u


    def forward(self, rgb, xyz, cls_ids,mask):

        xyzrgb = torch.cat([xyz, rgb], dim=1)
        ft, rs_xyz = self.xyznet(xyzrgb)
        b, c, h, w = ft.size()
        #print("ft.size()：", ft.size())

        px = self.prim_x(ft)
        #print("px shape:", px.shape)
        tx = self.trans(ft)
        sc = F.sigmoid(self.score(ft))
        # sc = self.score(ft)#这里去掉sigmoid

        #print("cls_ids 2:", cls_ids.shape)
        cls_ids = cls_ids.view(b).long()
        obj_ids = torch.tensor([i for i in range(b)]).long().cuda()
        px = px.view(b, -1, 4, h, w)[obj_ids, cls_ids]#变换后为64*4*32*32
        tx = tx.view(b, -1, 3, h, w)[obj_ids, cls_ids]#变换前为64*90*32*32
        sc = sc.view(b, -1, h, w)[obj_ids, cls_ids]#变换后为64*32*32
        del obj_ids

        # pr[bs, 4, h, w], tx[bs, 3, h, w], xyz[bs, 3, h, w]
        tx = tx + rs_xyz
        preds={'pred_r': px.contiguous(),
                'pred_t': tx.contiguous(),
                'pred_s': sc.contiguous(),
                'cls_id': cls_ids.contiguous()}
        return self.add_full(preds,mask) #模型的最终输出

class get_loss(nn.Module):#获取loss的类
    def __init__(self, dataset, scoring_weight=0.01, loss_type = "GADD", train = True,out_channel=32,):

        super(get_loss, self).__init__()
        self.prim_groups = dataset.prim_groups  # [obj_i:[gi:tensor[3, n]]]
        self.sym_list = dataset.get_sym_list()
        self.scoring_weight = scoring_weight
        self.loss_type = loss_type
        self.train = train
        self.select_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]#参与运算的物体id
        # self.full_net=get_full(9*32*32,3+out_channel*8)#输出通道，感觉也要同时估计sigmal,同时也要估计平移，这里也是模型的一部分，需要纳入训练及保存
        self.channel=out_channel
        
    def quaternion_matrix(self, pr):
        R = torch.cat(((1.0 - 2.0 * (pr[2, :] ** 2 + pr[3, :] ** 2)).unsqueeze(dim=1), \
                          (2.0 * pr[1, :] * pr[2, :] - 2.0 * pr[0, :] * pr[3, :]).unsqueeze(dim=1), \
                          (2.0 * pr[0, :] * pr[2, :] + 2.0 * pr[1, :] * pr[3, :]).unsqueeze(dim=1), \
                          (2.0 * pr[1, :] * pr[2, :] + 2.0 * pr[3, :] * pr[0, :]).unsqueeze(dim=1), \
                          (1.0 - 2.0 * (pr[1, :] ** 2 + pr[3, :] ** 2)).unsqueeze(dim=1), \
                          (-2.0 * pr[0, :] * pr[1, :] + 2.0 * pr[2, :] * pr[3, :]).unsqueeze(dim=1), \
                          (-2.0 * pr[0, :] * pr[2, :] + 2.0 * pr[1, :] * pr[3, :]).unsqueeze(dim=1), \
                          (2.0 * pr[0, :] * pr[1, :] + 2.0 * pr[2, :] * pr[3, :]).unsqueeze(dim=1), \
                          (1.0 - 2.0 * (pr[1, :] ** 2 + pr[2, :] ** 2)).unsqueeze(dim=1)),
                         dim=1).contiguous().view(-1, 3, 3)  # [nv, 3, 3]
        return R




    def calculate_ADD_or_ADDS(self, pred, gt_xyz, cls_id):

        if cls_id.item() in self.sym_list:

            num_valid, _, num_points = gt_xyz.size()
            inds = knn_one_point(pred.permute(0, 2, 1), gt_xyz.permute(0, 2, 1))  # num_valid, num_points
            inds = inds.view(num_valid, 1, num_points).repeat(1, 3, 1)
            tar_tmp = torch.gather(gt_xyz, 2, inds)
            add_ij = torch.mean(torch.norm(pred - tar_tmp, dim=1), dim=1)  # [nv]
        else:
            add_ij = torch.mean(torch.norm(pred - gt_xyz, dim=1), dim=1)  # [nv]

        return add_ij
        #以add为loss
    def loss_by_add(self,out_rot,gt_rot,out_trans,gt_trans,model_xyz,cls_id):
        _, _, num_points = model_xyz.shape
                        # print("model_xyz:", model_xyz[i].shape)
        b,c,_,_=out_rot.shape
        sub_value=[]
        md_xyz = model_xyz.view(-1,1, 3, num_points)

        # pt = pt.permute(1, 0).contiguous().unsqueeze(dim=2).repeat(1, 1, num_points)  # num_valid, 3, num_points
        pt=out_trans.view(b,1,3,1)
        # pred = torch.matmul(out_rot, md_xyz) + pt  # nv, 3, np
        pred = torch.matmul(out_rot, md_xyz) # nv, 3, np


        # t_tar = t_tar.contiguous().unsqueeze(dim=2).repeat(1, 1, num_points)
        # print("t_tar1:", t_tar.shape)
        t_tar=gt_trans.view(b,1,3,1)
        # gt_xyz = torch.matmul(gt_rot, md_xyz) + t_tar  # nv, 3, np
        gt_xyz = torch.matmul(gt_rot, md_xyz)  # nv, 3, np

        for i in range(b):
            # ADD(S)
            #32

            if cls_id[i] + 1 in self.select_id:
                
                sub_value.append(torch.mean(torch.norm(pred[i] - gt_xyz[i], dim=1), dim=1))

                # sub_loss_value[i] = torch.mean(add_ij * ps - self.scoring_weight * torch.log(ps))
            else:

                sub_value.append(add_ij * 0)

                # sub_loss_value[i] = torch.mean(add_ij * ps - self.scoring_weight * torch.log(ps)) * 0


                    # print('sub_loss_value:', sub_loss_value)
        return torch.stack(sub_value,dim=0)#64*32
#以adds为loss
    def loss_by_adds(self,out_rot,gt_rot,out_trans,gt_trans,model_xyz,cls_id):
        _, _, num_points = model_xyz.shape
                        # print("model_xyz:", model_xyz[i].shape)
        b,c,_,_=out_rot.shape
        sub_value=[]
        md_xyz = model_xyz.view(-1,1, 3, num_points)

        # pt = pt.permute(1, 0).contiguous().unsqueeze(dim=2).repeat(1, 1, num_points)  # num_valid, 3, num_points
        pt=out_trans.view(b,1,3,1)
        # pred = torch.matmul(out_rot, md_xyz) + pt  # nv, 3, np
        pred = torch.matmul(out_rot, md_xyz) # nv, 3, np


        # t_tar = t_tar.contiguous().unsqueeze(dim=2).repeat(1, 1, num_points)
        # print("t_tar1:", t_tar.shape)
        t_tar=gt_trans.view(b,1,3,1)
        # gt_xyz = torch.matmul(gt_rot, md_xyz) + t_tar  # nv, 3, np
        gt_xyz = torch.matmul(gt_rot, md_xyz)  # nv, 3, np

        for i in range(b):
            # ADD(S)
            #32

            if cls_id[i] + 1 in self.select_id:

                sub_value.append(self.calculate_ADD_or_ADDS(pred[i], gt_xyz[i], cls_id[i]))

                # sub_loss_value[i] = torch.mean(add_ij * ps - self.scoring_weight * torch.log(ps))
            else:

                sub_value.append(add_ij * 0)

                # sub_loss_value[i] = torch.mean(add_ij * ps - self.scoring_weight * torch.log(ps)) * 0


                    # print('sub_loss_value:', sub_loss_value)
        return torch.stack(sub_value,dim=0)#64*32
                    # gadd_loss = torch.mean(sub_loss_value)
    #使用分布的方式计算的损失
    def compute_Rloss(self,pre_r,gt_r,kappa):#64*32*3*3
        dot = torch.cosine_similarity(pre_r[:,:,:,:2], gt_r[:,:,:,:2], dim=2)
        dot=torch.clamp(dot,-0.99999,0.999999)
        loss_pixelwise = - torch.log(torch.square(kappa) + 1) \
                            + kappa * torch.acos(dot) \
                            + torch.log(1 + torch.exp(-kappa * np.pi))
        # loss_pixelwise =  torch.acos(dot)               
        loss_pixelwise = torch.where(torch.isinf(loss_pixelwise)|torch.isnan(loss_pixelwise), torch.full_like(loss_pixelwise, 0), loss_pixelwise)
        return loss_pixelwise
#当表示为四元数时计算loss
    def compute_Rloss_quter(self,pre_r,gt_r,kappa):#64*32*4
            #需要对pre_r进行归一化
            norms=torch.norm(pre_r,p=2,dim=-1,keepdim=True)
            eps=0.000001
            pre_r=pre_r/(norms+eps)
            dot = torch.cosine_similarity(pre_r[:,:,:], gt_r[:,:,:], dim=2)
            dot=torch.clamp(dot,-0.999,0.999)
            loss_pixelwise = - torch.log(torch.square(kappa) + 1) \
                                + kappa * torch.acos(dot) \
                                + torch.log(1 + torch.exp(-kappa * np.pi))
            # loss_pixelwise =  torch.acos(dot)               
            loss_pixelwise = torch.where(torch.isinf(loss_pixelwise)|torch.isnan(loss_pixelwise), torch.full_like(loss_pixelwise, 0), loss_pixelwise)
            return loss_pixelwise
    #直接计算角度误差
    def re_rad(self,R_est,R_gt):
        """Rotational Error.
        :param R_est: 3x3 tensor with the estimated rotation matrix.
        :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
        :return: The calculated error.
        """
        rotation_diff = torch.matmul(R_est,R_gt.transpose(3,2))
        trace = rotation_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
        trace=torch.clamp(trace,max=3)
        # Avoid invalid values due to numerical errors
        error_cos = torch.clamp(0.5 * (trace - 1.0), min=-0.99999, max=0.999999)
        rd_deg=torch.acos(error_cos)
        # rd_deg = np.rad2deg(np.arccos(error_cos))  #去掉这个弧度转度即可
        return rd_deg
    


    def forward(self, pre_t,pre_r,pre_s,pre_u, gt_r, gt_t, cls_ids, model_xyz, is_train=True,step=20):
        if self.train and self.loss_type!="Gaussian":
            pred_r = preds['pred_r']
            pred_t = preds['pred_t']
            pred_score = preds['pred_s']

            bs, c, h, w = pred_r.size()
            pred_r = pred_r.view(bs, 4, h, w)
            pred_r = pred_r / torch.norm(pred_r, dim=1, keepdim=True)#应该是归一化
            pred_r = pred_r.view(bs, 4, -1)
            pred_t = pred_t.view(bs, 3, -1)
            pred_score = pred_score.view(bs, -1)

            #print("cls-ids 3:", cls_ids.shape)
            cls_ids = cls_ids.view(bs)

            # for one batch
            mask = F.interpolate(mask, size=(h, w)).squeeze(dim=1)


            # add_lst = torch.zeros(bs).cuda()
            # add_loss_lst = torch.zeros(bs).cuda()
            #
            # gadd_lst = torch.zeros(bs).cuda()
            # gadd_loss_lst = torch.zeros(bs).cuda()

            sub_value = torch.zeros(bs).cuda()
            sub_loss_value = torch.zeros(bs).cuda()

            for i in range(bs):
                # get mask id
                mk = mask[i].view(-1)
                valid_pixels = mk.nonzero().view(-1)
                num_valid = valid_pixels.size()[0]
                if num_valid < 1:
                    continue
                if num_valid > 20:
                    selected = [i * step for i in range(int(num_valid / step))]
                    valid_pixels = valid_pixels[selected]
                    num_valid = valid_pixels.size()[0]

                # get r, t, s, cls
                pr = pred_r[i][:, valid_pixels]  # [4, nv]
                pt = pred_t[i][:, valid_pixels]  # [3, nv]
                ps = pred_score[i][valid_pixels]  # [nv]

                # rotation matrix
                R_pre = self.quaternion_matrix(pr)

                R_tar = gt_r[i].view(1, 3, 3).repeat(num_valid, 1, 1).contiguous()  # [nv, 3, 3]
                t_tar = gt_t[i].view(1, 3).repeat(num_valid, 1).contiguous()  # [nv, 3]


                if self.loss_type == "GADD":

                    # group
                    obj_grps = self.prim_groups[cls_ids[i]]
                    add_ij = torch.zeros((len(obj_grps), num_valid)).cuda()

                    for j, grp in enumerate(obj_grps):
                        _, num_p = grp.size()
                        grp = grp.cuda()
                        grp = grp.view(1, 3, num_p).repeat(num_valid, 1, 1)

                        npt = pt.permute(1, 0).contiguous().unsqueeze(dim=2).repeat(1, 1, num_p)  # [nv, 3, np]
                        ntt = t_tar.unsqueeze(dim=2).repeat(1, 1, num_p).contiguous()  # [nv, 3, np]


                        pred = torch.bmm(R_pre, grp) + npt  #

                        # pred = torch.bmm(R_pre, grp) + ntt  # rotation only

                        # pred = torch.bmm(R_tar, grp) + npt  # translation only

                        targ = torch.bmm(R_tar, grp) + ntt  # [nv, 3, np]

                        pred = pred.unsqueeze(dim=1).repeat(1, num_p, 1, 1).contiguous()  # [nv, np, 3, np]
                        targ = targ.permute(0, 2, 1).unsqueeze(dim=3).repeat(1, 1, 1, num_p).contiguous()  # [nv, np, 3, np]
                        min_dist, _ = torch.min(torch.norm(pred - targ, dim=2), dim=2)  # [nv, np]

                        if len(obj_grps) == 3 and j == 2:
                            # print('category 2')
                            ########################
                            add_ij[j] = torch.max(min_dist, dim=1)[0]  # [nv]
                        else:
                            add_ij[j] = torch.mean(min_dist, dim=1)  # [nv]

                    # ADD(S)
                    if len(obj_grps) == 3 and obj_grps[2].size()[1] > 1:
                        add_i = torch.max(add_ij, dim=0)[0]  # [nv]
                    else:
                        add_i = torch.mean(add_ij, dim=0)  # [nv]


                    # 指定训练物体
                    if cls_ids[i] + 1 in self.select_id:

                        # print('training {}'.format(cls_ids[i] + 1))

                        sub_value[i] = torch.mean(add_i)

                        sub_loss_value[i] = torch.mean(add_i * ps - self.scoring_weight * torch.log(ps))
                        # print('sub_loss_value {}: {}'.format(i, sub_loss_value[i]))

                    else:

                        sub_value[i] = torch.mean(add_i) * 0

                        sub_loss_value[i] = torch.mean(add_i * ps - self.scoring_weight * torch.log(ps)) * 0


                elif self.loss_type == "ADD":

                    # model
                    _, _, num_points = model_xyz.shape
                    # print("model_xyz:", model_xyz[i].shape)

                    md_xyz = torch.Tensor(model_xyz[i]).cuda().view(1, 3, num_points).repeat(num_valid, 1, 1)

                    pt = pt.permute(1, 0).contiguous().unsqueeze(dim=2).repeat(1, 1, num_points)  # num_valid, 3, num_points

                    pred = torch.bmm(R_pre, md_xyz) + pt  # nv, 3, np

                    t_tar = t_tar.contiguous().unsqueeze(dim=2).repeat(1, 1, num_points)
                    # print("t_tar1:", t_tar.shape)

                    gt_xyz = torch.bmm(R_tar, md_xyz) + t_tar  # nv, 3, np

                    # ADD(S)
                    add_ij = self.calculate_ADD_or_ADDS(pred, gt_xyz, cls_ids[i])

                    if cls_ids[i] + 1 in self.select_id:

                        sub_value[i] = torch.mean(add_ij)

                        sub_loss_value[i] = torch.mean(add_ij * ps - self.scoring_weight * torch.log(ps))

                    else:

                        sub_value[i] = torch.mean(add_ij) * 0

                        sub_loss_value[i] = torch.mean(add_ij * ps - self.scoring_weight * torch.log(ps)) * 0


            # print('sub_loss_value:', sub_loss_value)
            gadd = torch.mean(sub_value)
            gadd_loss = torch.mean(sub_loss_value)

            loss_dict = {'{}_loss'.format(self.loss_type): gadd_loss.item(), '{}'.format(self.loss_type): gadd.item()}


            # ignore the some sample with large outlier
            gadd_loss = torch.where(torch.isinf(gadd_loss), torch.full_like(gadd_loss, 0), gadd_loss)#这里应该也是一个数而不是向量叭
            gadd_loss = torch.where(torch.isnan(gadd_loss), torch.full_like(gadd_loss, 0), gadd_loss)


            return gadd_loss, loss_dict

        # elif self.train and self.loss_type=="Gaussian":
        elif self.loss_type=="Gaussian":
           
            bs, c, _= pre_r.size()
            #计算losssq
            #计算t的loss,解耦z和平移
            center_loss=F.l1_loss(pre_t[:,:2],gt_t[:,:2])
            z_loss=F.l1_loss(pre_t[:,2],gt_t[:,2])
            # z_loss=nn.L1Loss()
            #======训练时即约束=============
            R_matrix=ortho6d_to_mat_batch(pre_r)
            # #使用自中心
            # R_ego=pose_from_predictions_train(R_matrix,pre_t)
            R_ego=R_matrix
            #==========训练时并不约束==============
            # x_raw = pre_r[:,:, 0:3]  # bxc*3
            # y_raw = pre_r[:,:, 3:6]  # bxc*3
            # R_ego=torch.cat((x_raw.view(*x_raw.shape,1),y_raw.view(*y_raw.shape,1)),3)#b*c*3*2
            #======================
            R_tar=gt_r.view(bs,1, 3, 3).repeat(1,self.channel, 1, 1)
            #======计算R的loss--使用概率分布==========
            #=====使用旋转矩阵的方式=========
            R_loss=self.compute_Rloss(R_ego,R_tar,pre_u)#bs*c*2
            pre_s=pre_s.view(bs,self.channel,1)
            R_loss_mean0=torch.mean(torch.sum(pre_s*R_loss[:,:,:1],dim=1))
            R_loss_mean1=torch.mean(torch.sum(pre_s*R_loss[:,:,1:2],dim=1))
            # R_loss_mean0=torch.mean(R_loss[:,:,:1])
            # R_loss_mean1=torch.mean(R_loss[:,:,1:2])
            #求得最终R_loss
            R_loss_w=1*(R_loss_mean0+R_loss_mean1)
            #========使用四元数的方式============
            # R_tar=batch_rotation_matrix_to_quaternion(R_tar)
            # Rloss=self.compute_Rloss_quter(pre_r,R_tar,pre_u)
            # R_loss_w=torch.mean(torch.sum(pre_s*Rloss,dim=1))
            
            #==============计算R的loss方式二---使用add-s==================
            # pre_s=pre_s.squeeze()
            # R_loss_add=self.loss_by_adds(R_ego,R_tar,pre_t,gt_t,model_xyz,cls_ids)
            # R_loss_w=torch.mean(pre_s*R_loss_add)
            #=============计算Rloss的方式三---使用add=====================
            # pre_s=pre_s.squeeze()
            # R_loss_add=self.loss_by_add(R_ego,R_tar,pre_t,gt_t,model_xyz,cls_ids)
            # R_loss_w=torch.mean(torch.sum(pre_s*R_loss_add,dim=1))
            #===========bind loss================
            # R_loss=loss_by_bind(R_ego,R_tar,pre_t,gt_t)
            # R_loss=torch.mean(R_loss,dim=2)
            # R_loss_bind=torch.mean(pre_s*R_loss)
            #===========直接计算两个旋转之间的角度误差========
            # R_rad=self.re_rad(R_ego,R_tar)
            # R_rad_loss=torch.mean(torch.sum(pre_s*R_rad,dim=1))
            #可能需要排除无效值
            total_loss=center_loss+z_loss+ R_loss_w
            loss_dict = {'{}_loss'.format(self.loss_type): total_loss.item(),'center_loss':center_loss.item(),'z_loss':z_loss.item(),'R_loss': R_loss_w.item()}
            if is_train:
                return total_loss,loss_dict
            else:#是test则多返回几个
                #如果训练时不约束则需要在test时将其转为旋转举证
                # R_ego=ortho6d_to_mat_batch(pre_r)
                #====四元数转矩阵======
                # R_ego=quaternion_to_rotation_matrix(pre_r)
                return total_loss,loss_dict,pre_t,R_ego,pre_s.view(bs,self.channel,1),pre_u
        #  else:#test

        #     gadd_sub_value = torch.zeros(bs).cuda()
        #     gadd_sub_loss_value = torch.zeros(bs).cuda()

        #     add_sub_value = torch.zeros(bs).cuda()
        #     add_sub_loss_value = torch.zeros(bs).cuda()

        #     for i in range(bs):

        #         # get mask id
        #         mk = mask[i].view(-1)
        #         valid_pixels = mk.nonzero().view(-1)
        #         num_valid = valid_pixels.size()[0]
        #         if num_valid < 1:
        #             continue
        #         if num_valid > 20:
        #             selected = [i * step for i in range(int(num_valid / step))]#采样出一部分
        #             valid_pixels = valid_pixels[selected]
        #             num_valid = valid_pixels.size()[0]

        #         # get r, t, s, cls
        #         pr = pred_r[i][:, valid_pixels]  # [4, nv]
        #         pt = pred_t[i][:, valid_pixels]  # [3, nv]
        #         ps = pred_score[i][valid_pixels]  # [nv]

        #         # rotation matrix
        #         R_pre = self.quaternion_matrix(pr)#向量转矩阵

        #         R_tar = gt_r[i].view(1, 3, 3).repeat(num_valid, 1, 1).contiguous()  # [nv, 3, 3]
        #         t_tar = gt_t[i].view(1, 3).repeat(num_valid, 1).contiguous()  # [nv, 3]


        #         #  if self.loss_type == "GADD":

        #         obj_grps = self.prim_groups[cls_ids[i]]

        #         add_ij = torch.zeros((len(obj_grps), num_valid)).cuda()

        #         for j, grp in enumerate(obj_grps):
        #             _, num_p = grp.size()
        #             grp = grp.cuda()
        #             grp = grp.view(1, 3, num_p).repeat(num_valid, 1, 1)

        #             npt = pt.permute(1, 0).contiguous().unsqueeze(dim=2).repeat(1, 1, num_p)  # [nv, 3, np]
        #             ntt = t_tar.unsqueeze(dim=2).repeat(1, 1, num_p).contiguous()  # [nv, 3, np]
        #             pred = torch.bmm(R_pre, grp) + npt  # [nv, 3, np]，不太清楚这里为什么进行这样的转换
        #             targ = torch.bmm(R_tar, grp) + ntt  # [nv, 3, np]

        #             pred = pred.unsqueeze(dim=1).repeat(1, num_p, 1, 1).contiguous()  # [nv, np, 3, np]
        #             targ = targ.permute(0, 2, 1).unsqueeze(dim=3).repeat(1, 1, 1,
        #                                                                  num_p).contiguous()  # [nv, np, 3, np]
        #             min_dist, _ = torch.min(torch.norm(pred - targ, dim=2), dim=2)  # [nv, np]，选择最小的

        #             if len(obj_grps) == 3 and j == 2:
        #                 ########################
        #                 add_ij[j] = torch.max(min_dist, dim=1)[0]  # [nv]
        #             else:
        #                 add_ij[j] = torch.mean(min_dist, dim=1)  # [nv]

        #         # ADD(S)
        #         if len(obj_grps) == 3 and obj_grps[2].size()[1] > 1:
        #             add_i = torch.max(add_ij, dim=0)[0]  # [nv]

        #         else:
        #             add_i = torch.mean(add_ij, dim=0)  # [nv]


        #         sub_value[i] = torch.mean(add_i)
        #         sub_loss_value[i] = torch.mean(add_i * ps - self.scoring_weight * torch.log(ps))

        #         _, _, num_points = model_xyz.shape
        #         # print("model_xyz:", model_xyz[i].shape)

        #         md_xyz = torch.Tensor(model_xyz[i]).cuda().view(1, 3, num_points).repeat(num_valid, 1, 1)

        #         pt = pt.permute(1, 0).contiguous().unsqueeze(dim=2).repeat(1, 1, num_points)  # num_valid, 3, num_points
        #         pred = torch.bmm(R_pre, md_xyz) + pt  # nv, 3, np

        #         t_tar = t_tar.contiguous().unsqueeze(dim=2).repeat(1, 1, num_points)
        #         # print("t_tar1:", t_tar.shape)
        #         gt_xyz = torch.bmm(R_tar, md_xyz) + t_tar  # nv, 3, np
        #         # print("gt_xyz:", gt_xyz.shape)

        #         # ADD(S)
        #         add_ij = self.calculate_ADD_or_ADDS(pred, gt_xyz, cls_ids[i])
        #         # print("add_ij:", add_ij)

        #         add_sub_value[i] = torch.mean(add_ij)
        #         add_sub_loss_value[i] = torch.mean(add_ij * ps - self.scoring_weight * torch.log(ps))


        #     add = torch.mean(add_sub_value)#为什么使用的add是mean而不是min呢
        #     add_loss = torch.mean(add_sub_loss_value)

        #     gadd = torch.mean(gadd_sub_value)
        #     gadd_loss = torch.mean(gadd_sub_loss_value)


        #     loss_dict = {'add_loss': add_loss.item(), 'add': add.item(), 'gadd_loss': gadd_loss.item(), 'gadd': gadd.item()}
        #     # loss_dict = {'loss': add_loss.item(), 'add': add.item()}

        #     return add_loss, loss_dict





