
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

#加载数据
from dataset_loader_augment import train_dataset
#加载网络
from model.Fusion_net import FusionNet
#加载损失函数
import pytorch_losses


def Hybrid_Loss(pred, target, reduction='mean'):
    #先对输出做归一化处理
    pred = torch.sigmoid(pred)

    #BCE LOSS
    bce_loss = nn.BCELoss()
    bce_out = bce_loss(pred, target)

    #IOU LOSS
    iou_loss = pytorch_losses.IOU(reduction=reduction)
    iou_out = iou_loss(pred, target)

    #SSIM LOSS
    ssim_loss = pytorch_losses.SSIM(window_size=11)
    ssim_out = ssim_loss(pred, target)

    #Focal LOSS
    focal_losses = pytorch_losses.FocalLoss()
    focal_out = focal_losses(pred,target)

    #Dice LOSS
    dice_loss = pytorch_losses.DiceLoss()
    dice_out = dice_loss(pred, target)

    hybrid_loss = [bce_out, iou_out, ssim_out]
    losses = bce_out + iou_out + ssim_out

    return hybrid_loss, losses


def Multi_Loss(RGBD, RGB, depth, target, img_size, reduction='mean'):
    #先对输出做归一化处理
    RGBD = torch.sigmoid(RGBD)
    RGB = torch.sigmoid(RGB)
    depth = torch.sigmoid(depth)
    target = F.interpolate(target, img_size, mode='bilinear', align_corners=False)

    bce_loss = nn.BCELoss()
    iou_loss = pytorch_losses.IOU(reduction=reduction)
    ssim_loss = pytorch_losses.SSIM(window_size=11)

    #RGB loss
    RGB_loss = bce_loss(RGB, target) + iou_loss(RGB, target) + ssim_loss(RGB, target)
    RGBD_loss = bce_loss(RGBD, target) + iou_loss(RGBD, target) + ssim_loss(RGBD, target)
    depth_loss = bce_loss(depth, target) + iou_loss(depth, target) + ssim_loss(depth, target)

    hybrid_loss = [RGBD_loss, RGB_loss, depth_loss]
    losses = RGBD_loss + RGB_loss + depth_loss

    return hybrid_loss, losses


#获取当前学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#训练一个epoch
class Trainer(object):
    def __init__(self, cuda, model_fusion, optimizer, scheduler, train_loader, epochs, save_epoch):
        self.cuda = cuda
        self.model_fusion = model_fusion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.present_epoch = 0
        self.epochs = epochs          #总的epoch
        self.save_epoch = save_epoch    #间隔save_epoch保存一次权重文件

        self.train_loss_list = []
        self.train_log = './train_log.txt'
        with open(self.train_log, 'w') as f:
            f.write(('%10s' * 7) % ('Epoch', 'losses', 'loss_1', 'loss_2', 'loss_3', 'loss_4', 'lr'))
        f.close()


    def train_epoch(self):
        print(('\n' + '%10s' * 7) % ('Epoch', 'losses', 'loss_1', 'loss_2', 'loss_3', 'loss_4', 'lr'))
        # 初始化所有的loss
        losses_all, loss_1_all, loss_2_all, loss_3_all, loss_4_all = 0, 0, 0, 0, 0

        #设置进度条
        with tqdm(total=len(self.train_loader)) as pbar:
            for batch_idx, (img, mask, depth, edge) in enumerate(self.train_loader):

                if self.cuda:
                    img, mask, depth, edge = img.cuda(), mask.cuda(), depth.cuda(), edge.cuda()
                    img, mask, depth, edge = Variable(img), Variable(mask), Variable(depth), Variable(edge)

                n, c, h, w = img.size()  # batch_size, channels, height, weight
                #print(n,c,h,w)

                #梯度清零
                self.optimizer.zero_grad()

                depth = depth.view(n, 1, h, w)
                mask = mask.view(n, 1, h, w)
                edge = edge.view(n, 1, h, w)
                #前向传播
                Out_Fu1, Out_Fu2, Out_Fu3, Out_Fu4 = self.model_fusion(img, depth)

                #计算损失函数
                mask = mask.to(torch.float32)
                edge = edge.to(torch.float32)

                # loss_1 = F.binary_cross_entropy_with_logits(F1_out, mask, reduction='mean')
                # loss_2 = F.binary_cross_entropy_with_logits(F2_out, mask, reduction='mean')
                # loss_3 = F.binary_cross_entropy_with_logits(F3_out, mask, reduction='mean')
                # loss_4 = F.binary_cross_entropy_with_logits(F4_out, mask, reduction='mean')

                hybrid_loss_1, loss_1 = Hybrid_Loss(Out_Fu1, mask)
                hybrid_loss_2, loss_2 = Hybrid_Loss(Out_Fu2, mask)
                hybrid_loss_3, loss_3 = Hybrid_Loss(Out_Fu3, mask)
                hybrid_loss_4, loss_4 = Hybrid_Loss(Out_Fu4, mask)

                #计算损失函数用于反向传播
                loss_1_all = loss_1_all + loss_1.item()
                loss_2_all = loss_2_all + loss_2.item()
                loss_3_all = loss_3_all + loss_3.item()
                loss_4_all = loss_4_all + loss_4.item()
                losses = loss_1 + loss_2 + loss_3 + loss_4
                losses_all += losses.item()

                #实时更新信息
                s = ('%10s' * 1 + '%10.4g' * 6) % (self.present_epoch, losses.item(), loss_1.item(), loss_2.item(), loss_3.item(), loss_4.item(), get_lr(self.optimizer))
                pbar.set_description(s)
                pbar.update(1)

                #反向传播
                losses.backward()
                #更新权重
                self.optimizer.step()

        # 保存模型
        if self.present_epoch % self.save_epoch == 0:
            savename_ladder = ('checkpoint/RGBD-SOD_iter_%d.pth' % (self.present_epoch))
            torch.save(self.model_fusion.state_dict(), savename_ladder)

        #保存所有loss
        total_batch = len(self.train_loader)
        self.train_loss_list.append(losses_all / total_batch)

        #输出loss
        epoch_information = ('\n' + '%10s' * 1 + '%10.4g' * 6) % ((self.present_epoch), losses_all / total_batch,
        loss_1_all / total_batch, loss_2_all / total_batch, loss_3_all / total_batch, loss_4_all / total_batch,
        get_lr(self.optimizer))
        print(epoch_information)

        #写入文件
        with open(self.train_log, 'a') as f:
            f.write(epoch_information)
        f.close()

        #更新学习率
        self.scheduler.step()


    def draw_loss_plot(self):
        start_epoch = 0
        plt.figure()
        plt.plot(list(range(start_epoch, self.epochs)), self.train_loss_list, c='black', label='train_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('./loss.png')


    def train(self):
        for epoch in range(self.epochs):
            self.train_epoch()
            self.present_epoch += 1

        self.draw_loss_plot()


if __name__ == '__main__':
    """
        opt参数解析：
        train-dataset: 训练数据集路径
        img-size: 图片尺寸
        epochs: 训练总轮次
        batch-size: 批次大小
        workers: dataloader的最大worker数量
        save-epoch: 间隔save-epoch保存一次模型
        cuda: 是否使用GPU进行训练
        GPU-id: 使用单块GPU时设置的编号
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dataset', type=str, default='G:/dataset/RGB-D/train_dataset/train_2985/', help='path to the train dataset')
    parser.add_argument('--img-size', type=int, default=256, help='image size')
    #超参数设置
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs')
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate of learning rate')
    parser.add_argument('--save-epoch', type=int, default=1)
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda')
    parser.add_argument('--GPU-id', type=int, default=0)
    args = parser.parse_args()

    #加载训练集
    train_loader = torch.utils.data.DataLoader(train_dataset(args.train_dataset,image_size=args.img_size),batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)

    #加载模型
    model_fusion = FusionNet()
    #加载预训练权重
    ResNet_50_pretrained = './pretrained/resnet_50.pth'
    model_fusion.load_pretrained_model(ResNet_50_pretrained, ResNet_50_pretrained)

    # 使用GPU
    if args.cuda:
        assert torch.cuda.is_available, 'ERROR: cuda can not use'
        torch.cuda.set_device(args.GPU_id)  #指定显卡
        #torch.backends.cudnn.benchmark = True  # GPU网络加速
        model_fusion = model_fusion.cuda()
        #model_fusion = torch.nn.DataParallel(model_fusion)  #多GPU训练

    #定义优化器
    optimizer = optim.Adam(model_fusion.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 等间隔调整学习率，每训练step_size个epoch，lr*gamma
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    # 多间隔调整学习率，每训练至milestones中的epoch，lr*gamma
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 80], gamma=0.1)

    #开始训练
    training = Trainer(
        cuda=args.cuda,
        model_fusion=model_fusion,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        train_loader=train_loader,
        epochs=args.epochs,
        save_epoch=args.save_epoch
    )
    training.train()










