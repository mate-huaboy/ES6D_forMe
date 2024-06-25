import logging
import torch
import torch.nn.functional as F
import numpy as np
from lib.transformations import quaternion_matrix
import json
import pandas as pd
import pdb
from torch.optim import SGD



def ortho6d_to_mat_batch(poses):#poses：N*6，需要改成N*c*6的形式
    # poses bx6
        # poses
        x_raw = poses[:,:, 0:3]  # bx3
        y_raw = poses[:,:, 3:6]  # bx3

        # x = normalize_vector(x_raw)  # bx3
        x=F.normalize(x_raw,dim=2)
        z=torch.cross(x,y_raw)
        z=F.normalize(z,dim=2)
        y=torch.cross(z,x)
        # z = cross_product(x, y_raw)  # bx3
        # z = normalize_vector(z)  # bx3
        # y = cross_product(z, x)  # bx3

        x = x.view(*x.shape, 1)
        y = y.view(*y.shape, 1)
        z = z.view(*z.shape, 1)
        matrix = torch.cat((x, y, z), 3)  # bx3x3
        return matrix

def warnup_lr(cur_iter, end_iter, start_lr, end_lr):
    if(cur_iter < end_iter):
        return start_lr + (end_lr - start_lr) * cur_iter / end_iter
    else:
        return end_lr


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    l.addHandler(streamHandler)
    return l


def save_pred_and_gt_json(rt_list, total_instance_list, gt_rt_list, gt_cls_list, save_path):


    rt = np.stack(rt_list)
    gt = np.stack(gt_rt_list)
    cls = np.stack(gt_cls_list)

    jdict = {'pred_rt': rt.tolist(), 'gt_rt': gt.tolist(), 'cls': cls.tolist()}
    file_hd = open(save_path + '/results.json', 'w', encoding='utf-8')
    jobj = json.dump(jdict, file_hd)
    file_hd.close()

    scene_id_ls = []
    im_id_ls = []
    obj_id_ls = []
    score_ls = []
    time_ls = []

    for i in range (len(total_instance_list)):

        scene_id_ls.append(total_instance_list[i][0][0])
        im_id_ls.append(total_instance_list[i][0][1])
        obj_id_ls.append(cls.tolist()[i][0])

        score_ls.append(1)
        time_ls.append(-1)

    # exit()
    r_ls = []
    t_ls =[]

    for instance_pred in rt.tolist():

        instance_pred = np.array(instance_pred)
        pred_R = instance_pred[:, 0:3]

        # pred_R = pred_R.reshape((1, -1))[0].tolist()
        pred_R = pred_R.reshape((1, -1))[0].tolist()


        if len(pred_R) != 9:

            exit()

        pred_R = str(pred_R)[1:-1].replace(',', ' ')



        pred_T = instance_pred[:, 3:4].reshape((1, -1))[0] * 1000
        pred_T = pred_T.tolist()


        pred_T = str(pred_T)[1:-1].replace(',', ' ')


        r_ls.append(pred_R)
        t_ls.append(pred_T)

        # print("pred_R:", pred_R)
        # print("pred_T:", pred_T)

    # save_csv
    dataframe = pd.DataFrame({'scene_id': scene_id_ls, 'im_id': im_id_ls, 'obj_id': obj_id_ls, 'score': score_ls,
                            'R': r_ls, 't': t_ls, 'time': time_ls})
    dataframe.to_csv(save_path+ "/test.csv", index=False, sep=',')





def post_processing_ycb_1(preds, sym_list=[]):
    '''
    get final transform matrix T=[R|t] from prediction results with mask and score
    :param preds: output of pose net ['pred_x'][bs, 3, h, w]...
    :return: T[bs, 3, 4]
    '''
    cls_ids = preds['cls_id']
    b, c, h, w = preds['pred_x'].size()
    obj_ids = torch.tensor([i for i in range(b)]).long().cuda()
    sr = preds['score'].view(b, -1)
    st = preds['score'].view(b, -1)
    px = preds['pred_x'].view(b, 3, -1)
    py = preds['pred_y'].view(b, 3, -1)
    pz = preds['pred_z'].view(b, 3, -1)
    pt = preds['pred_t'].view(b, 3, -1)
    mask = preds['mask']

    mask = F.interpolate(mask, size=(h, w)).squeeze(dim=1).view(b, -1)
    res_T = []
    for i in range(b):
        valid_pixels = mask[i].nonzero().view(-1)
        num_val = valid_pixels.size()[0]
        if num_val < 32:
            valid_pixels = torch.ones(mask[i].size()).nonzero().view(-1)
            num_val = valid_pixels.size()[0]
        res_px = px[i][:, valid_pixels]
        res_py = py[i][:, valid_pixels]
        res_pz = pz[i][:, valid_pixels]
        res_pt = pt[i][:, valid_pixels]
        res_sr = sr[i][valid_pixels]
        res_st = st[i][valid_pixels]

        # res_px[3, nv] res_ux[nv]
        res_sr = torch.topk(res_sr, min(num_val, 32), dim=0, largest=True)
        res_st = torch.topk(res_st, min(num_val, 32), dim=0, largest=True)

        r_ids = res_sr.indices.unsqueeze(dim=0).repeat(3, 1)
        t_ids = res_st.indices.unsqueeze(dim=0).repeat(3, 1)

        res_sr = res_sr.values
        res_st = res_st.values

        res_px = torch.gather(res_px, dim=1, index=r_ids)
        res_py = torch.gather(res_py, dim=1, index=r_ids)
        res_pz = torch.gather(res_pz, dim=1, index=r_ids)
        res_pt = torch.gather(res_pt, dim=1, index=t_ids)
        # res_px[3, 32]
        res_px = torch.sum(res_sr * res_px, dim=1) / (torch.sum(res_sr) + 0.000001)
        res_py = torch.sum(res_sr * res_py, dim=1) / (torch.sum(res_sr) + 0.000001)
        res_pz = torch.sum(res_sr * res_pz, dim=1) / (torch.sum(res_sr) + 0.000001)
        res_pt = torch.sum(res_st * res_pt, dim=1) / (torch.sum(res_st) + 0.000001)
        res_sr = torch.sum(res_sr)
        res_st = torch.sum(res_st)

        res_px = res_px / torch.norm(res_px, dim=0).unsqueeze(dim=0)
        res_py = res_py / torch.norm(res_py, dim=0).unsqueeze(dim=0)
        res_pz = res_pz / torch.norm(res_pz, dim=0).unsqueeze(dim=0)


        if cls_ids[i].item() in sym_list:
            if cls_ids[i].item() == 12:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [2, 0, 1])
            if cls_ids[i].item() == 15:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [2, 0, 1])
            if cls_ids[i].item() == 18:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [1, 0, 2])
            if cls_ids[i].item() == 19:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [0, 1, 2])
            if cls_ids[i].item() == 20:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [1, 0, 2])
        else:
            res_r = primitives_to_rotation([res_px, res_py, res_pz], [2, 1, 0])
        res_T.append(torch.cat([res_r, res_pt.view(3, 1)], dim=1))

    return torch.stack(res_T, dim=0)


def post_processing_ycb_2(preds, sym_list=[]):
    '''
    get final transform matrix T=[R|t] from prediction results with mask and score
    :param preds: output of pose net ['pred_x'][bs, 3, h, w]...
    :return: T[bs, 3, 4]
    '''
    cls_ids = preds['cls_id']
    b, c, h, w = preds['pred_x'].size()
    obj_ids = torch.tensor([i for i in range(b)]).long().cuda()
    ux = preds['scor_x'].view(b, -1)
    uy = preds['scor_y'].view(b, -1)
    uz = preds['scor_z'].view(b, -1)
    ut = preds['scor_t'].view(b, -1)
    px = preds['pred_x'].view(b, 3, -1)
    py = preds['pred_y'].view(b, 3, -1)
    pz = preds['pred_z'].view(b, 3, -1)
    pt = preds['pred_t'].view(b, 3, -1)
    mask = preds['mask']

    mask = F.interpolate(mask, size=(h, w)).squeeze(dim=1).view(b, -1)
    res_T = []
    for i in range(b):
        valid_pixels = mask[i].nonzero().view(-1)
        num_val = valid_pixels.size()[0]
        if num_val < 32:
            valid_pixels = torch.ones(mask[i].size()).nonzero().view(-1)
            num_val = valid_pixels.size()[0]
        res_px = px[i][:, valid_pixels]
        res_py = py[i][:, valid_pixels]
        res_pz = pz[i][:, valid_pixels]
        res_pt = pt[i][:, valid_pixels]
        res_ux = ux[i][valid_pixels]
        res_uy = uy[i][valid_pixels]
        res_uz = uz[i][valid_pixels]
        res_ut = ut[i][valid_pixels]

        # res_px[3, nv] res_ux[nv]
        res_ux = torch.topk(res_ux, min(num_val, 32), dim=0, largest=True)
        res_uy = torch.topk(res_uy, min(num_val, 32), dim=0, largest=True)
        res_uz = torch.topk(res_uz, min(num_val, 32), dim=0, largest=True)
        res_ut = torch.topk(res_ut, min(num_val, 32), dim=0, largest=True)

        x_ids = res_ux.indices.unsqueeze(dim=0).repeat(3, 1)
        y_ids = res_uy.indices.unsqueeze(dim=0).repeat(3, 1)
        z_ids = res_uz.indices.unsqueeze(dim=0).repeat(3, 1)
        t_ids = res_ut.indices.unsqueeze(dim=0).repeat(3, 1)

        # res_ux = torch.mean(res_ux.values)
        # res_uy = torch.mean(res_uy.values)
        # res_uz = torch.mean(res_uz.values)
        # res_ut = torch.mean(res_ut.values)
        res_ux = res_ux.values
        res_uy = res_uy.values
        res_uz = res_uz.values
        res_ut = res_ut.values

        res_px = torch.gather(res_px, dim=1, index=x_ids)
        res_py = torch.gather(res_py, dim=1, index=y_ids)
        res_pz = torch.gather(res_pz, dim=1, index=z_ids)
        res_pt = torch.gather(res_pt, dim=1, index=t_ids)
        # res_px[3, 32]
        # res_px = torch.mean(res_px, dim=1)
        # res_py = torch.mean(res_py, dim=1)
        # res_pz = torch.mean(res_pz, dim=1)
        # res_pt = torch.mean(res_pt, dim=1)
        res_px = torch.sum(res_ux * res_px, dim=1) / (torch.sum(res_ux) + 0.000001)
        res_py = torch.sum(res_uy * res_py, dim=1) / (torch.sum(res_uy) + 0.000001)
        res_pz = torch.sum(res_uz * res_pz, dim=1) / (torch.sum(res_uz) + 0.000001)
        res_pt = torch.sum(res_ut * res_pt, dim=1) / (torch.sum(res_ut) + 0.000001)
        res_ux = torch.sum(res_ux)
        res_uy = torch.sum(res_uy)
        res_uz = torch.sum(res_uz)
        res_ut = torch.sum(res_ut)

        res_px = res_px / torch.norm(res_px, dim=0).unsqueeze(dim=0)
        res_py = res_py / torch.norm(res_py, dim=0).unsqueeze(dim=0)
        res_pz = res_pz / torch.norm(res_pz, dim=0).unsqueeze(dim=0)

        if cls_ids[i].item() in sym_list:
            if cls_ids[i].item() == 12:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [2, 0, 1])
            if cls_ids[i].item() == 15:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [2, 0, 1])
            if cls_ids[i].item() == 18:
                if res_ux > res_uz:
                    res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [1, 0, 2])
                else:
                    res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [1, 2, 0])
            if cls_ids[i].item() == 19:
                if res_uy > res_uz:
                    res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [0, 1, 2])
                else:
                    res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [0, 2, 1])
            if cls_ids[i].item() == 20:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [1, 0, 2])
        else:
            if res_ux >= res_uy and res_ux >= res_uz:
                if res_uy > res_uz:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [0, 1, 2])
                else:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [0, 2, 1])
            if res_uy >= res_ux and res_uy >= res_uz:
                if res_ux > res_uz:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [1, 0, 2])
                else:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [1, 2, 0])
            if res_uz >= res_ux and res_uz >= res_uy:
                if res_ux > res_uy:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [2, 0, 1])
                else:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [2, 1, 0])
        res_T.append(torch.cat([res_r, res_pt.view(3, 1)], dim=1))

    return torch.stack(res_T, dim=0)


def post_processing_ycb_3(preds, sym_list=[]):
    '''
    get final transform matrix T=[R|t] from prediction results with mask
    :param preds: output of pose net ['pred_x'][bs, 3, h, w]...
    :return: T[bs, 3, 4]
    '''
    cls_ids = preds['cls_id']
    b, c, h, w = preds['pred_x'].size()
    px = preds['pred_x'].view(b, 3, -1)
    py = preds['pred_y'].view(b, 3, -1)
    pz = preds['pred_z'].view(b, 3, -1)
    pt = preds['pred_t'].view(b, 3, -1)
    mask = preds['mask']

    mask = F.interpolate(mask, size=(h, w)).squeeze(dim=1).view(b, -1)
    res_T = []
    for i in range(b):
        valid_pixels = mask[i].nonzero().view(-1)
        num_val = valid_pixels.size()[0]
        if num_val < 32:
            valid_pixels = torch.ones(mask[i].size()).nonzero().view(-1)
            num_val = valid_pixels.size()[0]
        res_px = px[i][:, valid_pixels]
        res_py = py[i][:, valid_pixels]
        res_pz = pz[i][:, valid_pixels]
        res_pt = pt[i][:, valid_pixels]

        # get voting score
        res_ux = vote_strategy(res_px, dist_thr=0.05)
        res_uy = vote_strategy(res_py, dist_thr=0.05)
        res_uz = vote_strategy(res_pz, dist_thr=0.05)
        res_ut = vote_strategy(res_pt, dist_thr=0.025)

        # res_px[3, nv] res_ux[nv]
        res_ux = torch.topk(res_ux, min(num_val, 32), dim=0, largest=True)
        res_uy = torch.topk(res_uy, min(num_val, 32), dim=0, largest=True)
        res_uz = torch.topk(res_uz, min(num_val, 32), dim=0, largest=True)
        res_ut = torch.topk(res_ut, min(num_val, 32), dim=0, largest=True)

        x_ids = res_ux.indices.unsqueeze(dim=0).repeat(3, 1)
        y_ids = res_uy.indices.unsqueeze(dim=0).repeat(3, 1)
        z_ids = res_uz.indices.unsqueeze(dim=0).repeat(3, 1)
        t_ids = res_ut.indices.unsqueeze(dim=0).repeat(3, 1)

        res_ux = torch.mean(res_ux.values)
        res_uy = torch.mean(res_uy.values)
        res_uz = torch.mean(res_uz.values)
        res_ut = torch.mean(res_ut.values)

        res_px = torch.gather(res_px, dim=1, index=x_ids)
        res_py = torch.gather(res_py, dim=1, index=y_ids)
        res_pz = torch.gather(res_pz, dim=1, index=z_ids)
        res_pt = torch.gather(res_pt, dim=1, index=t_ids)
        # res_px[3, 32]
        res_px = torch.mean(res_px, dim=1)
        res_py = torch.mean(res_py, dim=1)
        res_pz = torch.mean(res_pz, dim=1)
        res_pt = torch.mean(res_pt, dim=1)

        res_px = res_px / torch.norm(res_px, dim=0).unsqueeze(dim=0)
        res_py = res_py / torch.norm(res_py, dim=0).unsqueeze(dim=0)
        res_pz = res_pz / torch.norm(res_pz, dim=0).unsqueeze(dim=0)

        if cls_ids[i].item() in sym_list:
            if cls_ids[i].item() == 12:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [2, 0, 1])
            if cls_ids[i].item() == 15:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [2, 0, 1])
            if cls_ids[i].item() == 18:
                if res_ux > res_uz:
                    res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [1, 0, 2])
                else:
                    res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [1, 2, 0])
            if cls_ids[i].item() == 19:
                if res_uy > res_uz:
                    res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [0, 1, 2])
                else:
                    res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [0, 2, 1])
            if cls_ids[i].item() == 20:
                res_r = primitives_to_rotation_sym([res_px, res_py, res_pz], [1, 0, 2])
        else:
            if res_ux >= res_uy and res_ux >= res_uz:
                if res_uy > res_uz:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [0, 1, 2])
                else:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [0, 2, 1])
            if res_uy >= res_ux and res_uy >= res_uz:
                if res_ux > res_uz:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [1, 0, 2])
                else:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [1, 2, 0])
            if res_uz >= res_ux and res_uz >= res_uy:
                if res_ux > res_uy:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [2, 0, 1])
                else:
                    res_r = primitives_to_rotation([res_px, res_py, res_pz], [2, 1, 0])
        res_T.append(torch.cat([res_r, res_pt.view(3, 1)], dim=1))

    return torch.stack(res_T, dim=0)


def primitives_to_rotation_sym(prim_list, order_list=[0, 1, 2]):
    p = []
    p.append(prim_list[order_list[0]])
    p.append(prim_list[order_list[1]])
    p[1] = p[1] - torch.dot(p[0], p[1]) * p[0]
    p[1] = p[1] / torch.norm(p[1])
    p3 = torch.zeros(p[1].size()).cuda().float()
    p3[0] = p[0][1] * p[1][2] - p[0][2] * p[1][1]
    p3[1] = - p[0][0] * p[1][2] + p[0][2] * p[1][0]
    p3[2] = p[0][0] * p[1][1] - p[0][1] * p[1][0]
    p.append(p3)

    if order_list[0] == 0 and order_list[1] == 1:
        px = p[0]
        py = p[1]
        pz = p[2]
    if order_list[0] == 0 and order_list[1] == 2:
        px = p[0]
        py = -p[2]
        pz = p[1]
    if order_list[0] == 1 and order_list[1] == 0:
        px = p[1]
        py = p[0]
        pz = -p[2]
    if order_list[0] == 1 and order_list[1] == 2:
        px = p[2]
        py = p[0]
        pz = p[1]
    if order_list[0] == 2 and order_list[1] == 0:
        px = p[1]
        py = p[2]
        pz = p[0]
    if order_list[0] == 2 and order_list[1] == 1:
        px = -p[2]
        py = p[1]
        pz = p[0]

    res_R = torch.cat([px.unsqueeze(dim=1), py.unsqueeze(dim=1), pz.unsqueeze(dim=1)], dim=1)
    return res_R


def primitives_to_rotation(prim_list, order_list=[0, 1, 2]):
    AA = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    m = AA.shape[1]
    BB = torch.stack(prim_list, dim=0).detach().cpu().numpy().astype(np.float32)
    # rotation matirx
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)
    return torch.from_numpy(R).cuda()


def vote_strategy(pred_x, dist_thr=0.1):
    c, n = pred_x.size()
    # 1. calculate distance between every pred_x[i] and pred_x[j]
    x1 = pred_x.unsqueeze(dim=0).repeat(n, 1, 1)
    x2 = pred_x.permute(1, 0).unsqueeze(dim=2).repeat(1, 1, n)
    d = torch.norm(x1 - x2, dim=1)
    # 2. count the vote of pred_x[i] by distance and the threshold
    d = d < dist_thr
    vote_num = torch.sum(d, dim=1).float()
    return vote_num


def cal_mean_std(dataset):
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                              num_workers=0, pin_memory=True)

    mean = torch.zeros(3)
    std = torch.zeros(3)

    print('==> Computing mean and std..')

    print("len dataset:", len(dataset))

    mask_0 = []

    for data in trainloader:

        inputs = data['rgb']# .cuda()
        # print("mask_flag:", data['mask_flag'])
        if data['mask_flag'] != 0:
            print("mask_flag:", data['mask_flag'])

            mask_0.append(data['mask_flag'][0])

        # print("len dataset:", len(dataset))
        #
        # print("inputs size:", inputs.shape)
        # exit()

        for i in range(3):

            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()

    mean.div_(len(dataset))
    std.div_(len(dataset))

    # print('mean: {}, std: {}'.format(mean, std))

    return mean, std, mask_0

def post_processing_minloss(R_matrix,pre_s,kappa):
    #确定初始值
   
    # max_values, max_indices = torch.max(pre_s, dim=1, keepdim=True)
    x1=R_matrix[:,:,:,:2]
    # x1 = selected_R_values[:,:,:2].unsqueeze(1).requires_grad_(True)#64*32*3*2
    x2=R_matrix[:,:,:,:2]
    x1=x1.unsqueeze(2)
    x2=x2.unsqueeze(1)
    dot=torch.cosine_similarity(x1, x2, dim=3)
    dot=torch.clamp(dot,-0.999999,0.999999)
    kappa=kappa.unsqueeze(1)
    loss_pixelwise = - torch.log(torch.square(kappa) + 1) \
                        + kappa * torch.acos(dot) \
                        + torch.log(1 + torch.exp(-kappa * np.pi))
    # loss_pixelwise = kappa * torch.acos(dot) 
    loss_pixelwise = torch.where(torch.isinf(loss_pixelwise)|torch.isnan(loss_pixelwise), torch.full_like(loss_pixelwise, 0), loss_pixelwise)
    pre_s=pre_s.unsqueeze(1)
    R_loss_mean0=torch.sum(pre_s*loss_pixelwise[:,:,:,:1],dim=2)
    R_loss_mean1=torch.sum(pre_s*loss_pixelwise[:,:,:,1:2],dim=2)
    #求得最终R_loss
    R_loss=R_loss_mean0+R_loss_mean1
    min_values,min_indices=torch.min(R_loss,dim=1)
    res= R_matrix[range(R_matrix.size(0)),min_indices.squeeze(),:,:].squeeze()#64*3*3
    return res


def post_processing_optimization(R_matrix,pre_s,kappa,num_iterations):
    #确定初始值
    with torch.enable_grad():
        max_values, max_indices = torch.max(pre_s, dim=1, keepdim=True)
        # selected_R_values=R_matrix[]
        # 使用索引从 R_matrix 中选择对应的值
        selected_R_values = R_matrix[range(R_matrix.size(0)),max_indices.squeeze(),:,:].squeeze()#64*3*3

        x1 = selected_R_values[:,:,:2].unsqueeze(1).requires_grad_(True)#64*3*2
        learning_rate = 100
        optimizer = SGD([x1], lr=learning_rate)
        # def object_func(mu,w,kappa):
        #     term1 = (kappa[:, 0]**2 + 1) * torch.exp(-kappa[:, 0] * torch.acos(x1.matmul(mu[:, 0])))
        #     term2 = (1 + torch.exp(-kappa[:, 0] * torch.pi))
        #     term3 = (kappa[:, 1]**2 + 1) * torch.exp(-kappa[:, 1] * torch.acos(x2.matmul(mu[:, 1])))
        #     term4 = (1 + torch.exp(-kappa[:, 1] * torch.pi))
            
        #     return torch.sum(w * (term1 / term2) * (term3 / term4))
    

        optimizer.zero_grad()
        for iteration in range(num_iterations):
            # loss = -objective_function(x1, x2, pre_s, kappa, mu)  # 使用负号求最大值相当于求最小值
            #求loss
            dot=torch.cosine_similarity(R_matrix[:,:,:,:2], x1, dim=2)
            dot=torch.clamp(dot,-0.999999,0.999999)
            loss_pixelwise = - torch.log(torch.square(kappa) + 1) \
                                + kappa * torch.acos(dot) \
                                + torch.log(1 + torch.exp(-kappa * np.pi))
            # loss_pixelwise = kappa * torch.acos(dot) 
            loss_pixelwise = torch.where(torch.isinf(loss_pixelwise)|torch.isnan(loss_pixelwise), torch.full_like(loss_pixelwise, 0), loss_pixelwise)
            R_loss_mean0=torch.mean(torch.sum(pre_s*loss_pixelwise[:,:,:1],dim=1))
            R_loss_mean1=torch.mean(torch.sum(pre_s*loss_pixelwise[:,:,1:2],dim=1))
            #求得最终R_loss
            R_loss=R_loss_mean0+R_loss_mean1
            R_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            mtrix=ortho6d_to_mat_batch(x1.permute(0,1,3,2).reshape(-1, 1, 6))
            x1.data=mtrix[:,:,:,:2].view_as(x1).data

            # 打印每100次迭代的损失
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {-loss.item()}")
        return mtrix.squeeze().detach()
    # 打印最终结果
    # print("Optimized x1:")
    # print(x1.detach().numpy())
    # print("Optimized x2:")
    # print(x2.detach().numpy())

def post_processing_max(R_matrix,pre_s):

    max_values, max_indices = torch.max(pre_s, dim=1, keepdim=True)
    # selected_R_values=R_matrix[]
    # 使用索引从 R_matrix 中选择对应的值
    selected_R_values = R_matrix[range(R_matrix.size(0)),max_indices.squeeze(),:,:]
    
    return selected_R_values

def post_processing_ycb_quaternion(preds, sym_list=[]):
    '''
    get final transform matrix T=[R|t] from prediction results
    :return: T[bs, 3, 4]
    '''
    cls_ids = preds['cls_id']
    b, c, h, w = preds['pred_r'].size()
    px = preds['pred_r'].detach().view(b, 4, -1)
    pt = preds['pred_t'].detach().view(b, 3, -1)
    ps = preds['pred_s'].detach().view(b, -1)
    mask = preds['xyz'][:, 0].unsqueeze(dim=1).detach()

    mask = F.interpolate(mask, size=(h, w)).squeeze(dim=1).view(b, -1)
    res_T = []
    for i in range(b):
        valid_pixels = mask[i].nonzero().view(-1)
        num_val = valid_pixels.size()[0]
        if num_val < 32:
            valid_pixels = torch.ones(mask[i].size()).nonzero().view(-1)
            num_val = valid_pixels.size()[0]
        q = px[i].view(4, -1)[:, valid_pixels]
        t = pt[i].view(3, -1)[:, valid_pixels]
        s = ps[i].view(-1)[valid_pixels]
        s_id = torch.argmax(s)#选择最大的哪一个
        _q = q[:, s_id].cpu().numpy()
        _r = quaternion_matrix(_q)[:3, :3]
        _r = torch.from_numpy(_r).cuda().float()
        _t = t[:, s_id].view(3, 1)
        res_T.append(torch.cat([_r, _t], dim=1))

    return torch.stack(res_T, dim=0)
def quat2mat_torch(quat, eps=0.0):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: [B, 4]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    assert quat.ndim == 2 and quat.shape[1] == 4, quat.shape
    norm_quat = quat.norm(p=2, dim=1, keepdim=True)
    # print('quat', quat) # Bx4
    # print('norm_quat: ', norm_quat)  # Bx1
    norm_quat = quat / (norm_quat + eps)
    # print('normed quat: ', norm_quat)
    qw, qx, qy, qz = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    B = quat.size(0)

    s = 2.0  # * Nq = qw*qw + qx*qx + qy*qy + qz*qz
    X = qx * s
    Y = qy * s
    Z = qz * s
    wX = qw * X
    wY = qw * Y
    wZ = qw * Z
    xX = qx * X
    xY = qx * Y
    xZ = qx * Z
    yY = qy * Y
    yZ = qy * Z
    zZ = qz * Z
    rotMat = torch.stack(
        [1.0 - (yY + zZ), xY - wZ, xZ + wY, xY + wZ, 1.0 - (xX + zZ), yZ - wX, xZ - wY, yZ + wX, 1.0 - (xX + yY)], dim=1
    ).reshape(B, 3, 3)

    # rotMat = torch.stack([
    #     qw * qw + qx * qx - qy * qy - qz * qz, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
    #     2 * (qx * qy + qw * qz), qw * qw - qx * qx + qy * qy - qz * qz, 2 * (qy * qz - qw * qx),
    #     2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz],
    #     dim=1).reshape(B, 3, 3)

    # w2, x2, y2, z2 = qw*qw, qx*qx, qy*qy, qz*qz
    # wx, wy, wz = qw*qx, qw*qy, qw*qz
    # xy, xz, yz = qx*qy, qx*qz, qy*qz

    # rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
    #                       2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
    #                       2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat
#旋转矩阵转四元数
#从pytorch3d中拿的代码
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def batch_rotation_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    # pyre-ignore [16]: `torch.Tensor` has no attribute `new_tensor`.
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(q_abs.new_tensor(0.1)))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(*batch_dim, 4)

def batch_rotation_matrix_to_quaternion_discard(rotation_matrices):
    r = rotation_matrices
    q = torch.zeros(rotation_matrices.size()[:-2] + (4,), device=rotation_matrices.device)
    
    trace = r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]
    
    mask = trace > 0#解决存在nan值的问题
    mask1=trace>-1
    q[...,0]=torch.where(mask1,torch.sqrt(1 + torch.clamp(trace, min=-1)) / 2,torch.ones_like(q[...,0]))
    # q[..., 0] = torch.sqrt(1 + torch.clamp(trace, min=-1)) / 2
    q[..., 1] = torch.where(mask, (r[..., 2, 1] - r[..., 1, 2]) / (4 * q[..., 0]), torch.zeros_like(q[..., 0]))
    q[..., 2] = torch.where(mask, (r[..., 0, 2] - r[..., 2, 0]) / (4 * q[..., 0]), torch.zeros_like(q[..., 0]))
    q[..., 3] = torch.where(mask, (r[..., 1, 0] - r[..., 0, 1]) / (4 * q[..., 0]), torch.zeros_like(q[..., 0]))
    
    return q



def allo_to_ego_mat_torch(translation, rot_allo, eps=1e-4):
    # translation: Nx3
    # rot_allo: Nxc*3x3
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = torch.tensor([0, 0, 1.0], dtype=translation.dtype, device=translation.device)  # (3,)
    obj_ray = translation / (torch.norm(translation, dim=1, keepdim=True) + eps)

    # cam_ray.dot(obj_ray), assume cam_ray: (0, 0, 1)
    angle = obj_ray[:, 2:3].acos()

    # Compute rotation between ray to object centroid and optical center ray
    axis = torch.cross(cam_ray.expand_as(obj_ray), obj_ray)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + eps)

    # Build quaternion representing the rotation around the computed axis
    # angle-axis => quat
    q_allo_to_ego = torch.cat(
        [
            torch.cos(angle / 2.0),
            axis[:, 0:1] * torch.sin(angle / 2.0),
            axis[:, 1:2] * torch.sin(angle / 2.0),
            axis[:, 2:3] * torch.sin(angle / 2.0),
        ],
        dim=1,
    )
    rot_allo_to_ego = quat2mat_torch(q_allo_to_ego)#N*3*3
    rot_allo_to_ego=rot_allo_to_ego.unsqueeze(1)
    # Apply quaternion for transformation from allocentric to egocentric.
    rot_ego = torch.matmul(rot_allo_to_ego, rot_allo)
    return rot_ego

def pose_from_predictions_train(
    pred_rots,
    pred_trans,
    eps=1e-4,
    is_allo=True,
):
    """for train
    Args:
        pred_rots:
        pred_trans
        eps:
        is_allo:
        z_type: REL | ABS | LOG | NEG_LOG

    Returns:

    """
   
    if pred_rots is not None:
        if pred_rots.ndim == 2 and pred_rots.shape[-1] == 4:
            pred_quats = pred_rots
            quat_allo = pred_quats / (torch.norm(pred_quats, dim=1, keepdim=True) + eps)
            if is_allo:
                quat_ego = allocentric_to_egocentric_torch(translation, quat_allo, eps=eps)
            else:
                quat_ego = quat_allo
            rot_ego = quat2mat_torch(quat_ego)
        if pred_rots.ndim == 4 and pred_rots.shape[-1] == 3:  # N*cx3x3
            if is_allo:
                rot_ego = allo_to_ego_mat_torch(pred_trans, pred_rots, eps=eps)
            else:
                rot_ego = pred_rots
    else:
        rot_ego=None
    return rot_ego

#四元数转旋转向量
def quaternion_to_rotation_vector(quaternions):
    # 将四元数归一化
    quaternions = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)
    
    # 计算旋转角度
    angle = 2 * torch.acos(quaternions[..., 0])
    
    # 计算旋转轴
    axis = quaternions[..., 1:] / torch.sqrt(1 - quaternions[..., 0].pow(2))
    
    # 将旋转轴和角度转换为旋转向量
    rotation_vector = angle.unsqueeze(-1) * axis
    
    return rotation_vector
#四元数转旋转矩阵
def quaternion_to_rotation_matrix(quaternions):
    # 将四元数归一化
    quaternions = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)
    
    # 提取四元数的各个分量
    w, x, y, z = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    
    # 计算旋转矩阵的各个元素
    rotation_matrix =  torch.stack([
        torch.stack([1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w], dim=-1),
        torch.stack([2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w], dim=-1),
        torch.stack([2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2], dim=-1)
    ], dim=-2)
    
    return rotation_matrix


def post_processing_ycb_quaternion_wi_vote(preds, sym_list=[]):
    '''
    get final transform matrix T=[R|t] from prediction results with mask
    :param preds: output of pose net ['pred_x'][bs, 4, h, w]...
    :return: T[bs, 3, 4]
    '''
    cls_ids = preds['cls_id']
    b, c, h, w = preds['pred_r'].size()
    px = preds['pred_r'].detach().view(b, 4, -1)
    pt = preds['pred_t'].detach().view(b, 3, -1)

    ps = preds['pred_s'].detach().view(b, -1)   # confidence score

    mask = preds['xyz'][:, 0].unsqueeze(dim=1).detach()

    mask = F.interpolate(mask, size=(h, w)).squeeze(dim=1).view(b, -1)
    res_T = []

    for i in range(b):

        valid_pixels = mask[i].nonzero().view(-1)
        num_val = valid_pixels.size()[0]

        if num_val < 32:

            valid_pixels = torch.ones(mask[i].size()).nonzero().view(-1)
            num_val = valid_pixels.size()[0]

        q = px[i].view(4, -1)[:, valid_pixels]
        t = pt[i].view(3, -1)[:, valid_pixels]
        s = ps[i].view(-1)[valid_pixels]


        k_s = torch.topk(s, min(num_val, 32), dim=0, largest=True)
        s_id = k_s.indices.unsqueeze(dim=0).repeat(4, 1)
        s_v = k_s.values
        res_t = torch.gather(t, dim=1, index=s_id[:3, :])
        res_q = torch.gather(q, dim=1, index=s_id[:4, :])
        n_id = res_q[0, :] < 0
        res_q[:, n_id] = -res_q[:, n_id]


        # res_px[3, 32]
        res_t = torch.sum(s_v * res_t, dim=1) / max(torch.sum(s_v), 0.0001)
        res_q = torch.sum(s_v * res_q, dim=1) / max(torch.sum(s_v), 0.0001)
        res_q = res_q / torch.norm(res_q, dim=0)
        s_id = torch.argmax(s)

        _q = q[:, s_id].cpu().numpy()
        # _q = res_q.cpu().numpy()

        _r = quaternion_matrix(_q)[:3, :3]
        _r = torch.from_numpy(_r).cuda().float()
        _t = res_t.view(3, 1)
        res_T.append(torch.cat([_r, _t], dim=1))

    return torch.stack(res_T, dim=0)


def post_processing_translation_and_ratation(preds, sym_list=[]):
    '''
    get final transform matrix T=[R|t] from prediction results with mask
    :param preds: output of pose net ['pred_x'][bs, 4, h, w]...
    :return: T[bs, 3, 4]
    '''
    cls_ids = preds['cls_id']
    b, c, h, w = preds['pred_r'].size()
    px = preds['pred_r'].detach().view(b, 4, -1)
    pt = preds['pred_t'].detach().view(b, 3, -1)
    r_score = preds['pred_rs'].detach().view(b, -1)
    t_score = preds['pred_ts'].detach().view(b, -1)
    mask = preds['xyz'][:, 0].unsqueeze(dim=1).detach()

    mask = F.interpolate(mask, size=(h, w)).squeeze(dim=1).view(b, -1)
    res_T = []
    for i in range(b):
        valid_pixels = mask[i].nonzero().view(-1)
        num_val = valid_pixels.size()[0]
        if num_val < 32:
            valid_pixels = torch.ones(mask[i].size()).nonzero().view(-1)
            num_val = valid_pixels.size()[0]
        q = px[i].view(4, -1)[:, valid_pixels]
        t = pt[i].view(3, -1)[:, valid_pixels]
        rs = r_score[i].view(-1)[valid_pixels]
        ts = t_score[i].view(-1)[valid_pixels]
        rs_id = torch.argmax(rs)
        ts_id = torch.argmax(ts)
        _q = q[:, rs_id].cpu().numpy()
        _r = quaternion_matrix(_q)[:3, :3]
        _r = torch.from_numpy(_r).cuda().float()
        _t = t[:, ts_id].view(3, 1)
        res_T.append(torch.cat([_r, _t], dim=1))

    return torch.stack(res_T, dim=0)
