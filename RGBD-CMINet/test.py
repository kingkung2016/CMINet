
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from dataset_loader_augment import test_dataset
from saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm
#加载网络
from model.Fusion_net import FusionNet

#加载测试图片和真实图片
class calculate_dataset:
    def __init__(self, image_root, gt_root):
        self.img_list = [os.path.splitext(f)[0] for f in os.listdir(gt_root) if f.endswith('.png')]
        self.image_root = image_root
        self.gt_root = gt_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.img_list)
        self.index = 0

    def load_data(self):
        #根据不同论文，需要修改图片命名格式
        image = self.binary_loader(os.path.join(self.image_root,self.img_list[self.index]+ '.png'))
        #测试集所有mask都是png格式
        gt = self.binary_loader(os.path.join(self.gt_root,self.img_list[self.index] + '.png'))
        self.index += 1
        return image, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


def test(test_root, model_fusion, salmap_root, image_size, cuda):
    #加载测试集
    test_loader = torch.utils.data.DataLoader(test_dataset(test_root, image_size),
                                   batch_size=1, shuffle=True, num_workers=0, pin_memory=True,drop_last=False)
    #设置为eval
    model_fusion.eval()
    #进行测试
    for id, (data, depth, img_name, img_size) in enumerate(test_loader):
        # print('testing bach %d' % id)
        if cuda:
            inputs = Variable(data).cuda()
            depth = Variable(depth).cuda()
        n, c, h, w = inputs.size()
        #print(n,c,h,w)

        #生成测试结果
        F_out, F2_out, F3_out, F4_out = model_fusion(inputs, depth)
        pred = torch.sigmoid(F_out)
        outputs = pred[0, 0].detach().cpu().numpy()

        #保存测试图片
        plt.imsave(os.path.join(salmap_root, img_name[0] + '.png'), outputs, cmap='gray')

        #保存所有阶段图片
        stage1_pred = torch.sigmoid(F_out)[0, 0].detach().cpu().numpy()
        # stage2_pred = torch.sigmoid(F2_out)[0, 0].detach().cpu().numpy()
        # stage3_pred = torch.sigmoid(F3_out)[0, 0].detach().cpu().numpy()
        # stage4_pred = torch.sigmoid(F4_out)[0, 0].detach().cpu().numpy()

        plt.imsave(os.path.join(salmap_root, img_name[0] + '.png'), stage1_pred, cmap='gray')
        # plt.imsave(os.path.join(salmap_root, img_name[0] + '_stage2.png'), stage2_pred, cmap='gray')
        # plt.imsave(os.path.join(salmap_root, img_name[0] + '_stage3.png'), stage3_pred, cmap='gray')
        # plt.imsave(os.path.join(salmap_root, img_name[0] + '_stage4.png'), stage4_pred, cmap='gray')



    #计算评价指标
    mask_root = test_root + '/test_masks/'
    dataset_name = test_root.split('/')[-1]
    #加载数据集
    test_loader = calculate_dataset(salmap_root, mask_root)
    #定义评价指标
    mae, fm, sm, em, wfm = cal_mae(), cal_fm(test_loader.size), cal_sm(), cal_em(), cal_wfm()
    res = []
    for i in tqdm(range(test_loader.size)):
        sal, gt = test_loader.load_data()
        # 尺寸不一致则修改尺寸
        if sal.size != gt.size:
            x, y = sal.size
            gt = gt.resize((x, y))

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        gt[gt > 0.5] = 1
        gt[gt != 1] = 0
        res = sal
        res = np.array(res)
        if res.max() == res.min():
            res = res / 255
        else:
            res = (res - res.min()) / (res.max() - res.min())
        mae.update(res, gt)
        sm.update(res, gt)
        fm.update(res, gt)
        em.update(res, gt)
        wfm.update(res, gt)

    MAE = mae.show()
    maxf, meanf, _, _ = fm.show()
    sm = sm.show()
    em = em.show()
    wfm = wfm.show()
    #评价指标分别是MAE,maxF,avgF,加权F,S,E
    print('\n{}:  MAE: {:.4f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f}'
          .format(dataset_name, MAE, maxf,meanf, wfm, sm,em))

if __name__ == '__main__':
    """
        opt参数解析：
        ckpt: 权重文件的路径
        img-size: 图片尺寸
        test-dataroot: 测试集的路径
        salmap-root: 预测图片保存路径
        cuda: 是否使用GPU进行训练
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', type=int, default=256, help='image size')
    parser.add_argument('--ckpt', type=str, default='./checkpoint/RGBD-SOD_iter_99.pth', help='num of checkpoints')
    parser.add_argument('--test-root', type=str, default='/media/ubuntu/新加卷/Kang Yi/dataset/RGB-D/test_dataset/', help='path to the test dataset')
    parser.add_argument('--salmap-root', type=str, default='./sal_map/', help='path to saliency map')
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda')
    args = parser.parse_args()

    #定义测试集
    test_data = ['DES-RGBD135', 'NLPR', 'DUT-RGBD', 'NJUD', 'STERE-797', 'STERE-1000', 'SSD', 'LFSD', 'SIP']

    # 加载三个网络
    model_fusion = FusionNet()
    # 加载权重文件
    model_fusion.load_state_dict(torch.load(args.ckpt, map_location='cuda:0'))

    # 使用GPU
    if args.cuda:
        assert torch.cuda.is_available, 'ERROR: cuda can not use'
        #torch.backends.cudnn.benchmark = True  # GPU网络加速
        model_fusion = model_fusion.cuda()

    #定义测试集
    for test_name in test_data:
        test_root = os.path.join(args.test_root + test_name)
        #存储测试图片的路径
        save_salmap_root = os.path.join(args.salmap_root + test_name)
        if os.path.exists(save_salmap_root):
            shutil.rmtree(save_salmap_root)
        os.mkdir(save_salmap_root)
        test(test_root, model_fusion, save_salmap_root, args.img_size, args.cuda)











