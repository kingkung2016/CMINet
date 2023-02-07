
import numpy as np
import torch
import time

if __name__ == '__main__':
    #卷积加速
    torch.backends.cudnn.benchmark = True

    # 加载网络
    from model.Fusion_net import FusionNet
    #加载模型
    model_fusion = FusionNet().cuda()
    #加载预训练权重
    ResNet_50_pretrained = './pretrained/resnet_50.pth'
    model_fusion.load_pretrained_model(ResNet_50_pretrained, ResNet_50_pretrained)

    #测试模型
    model_fusion = model_fusion.eval()

    #查看网络结构
    from torchsummaryX import summary
    summary(model_fusion, torch.zeros(1, 3, 256, 256).cuda(), torch.zeros(1, 1, 256, 256).cuda())

    from DNN_printer import DNN_printer
    DNN_printer(model_fusion, input_size=[(3,256,256), (1,256,256)], batch_size=1)

    #计算速度
    img = torch.randn(1, 3, 256, 256).cuda()
    depth = torch.randn(1, 1, 256, 256).cuda()

    time_spent = []
    for idx in range(100):
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            _, _, _, _ = model_fusion(img, depth)

        torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        end = time.time()

        if idx > 10:
            time_spent.append(end - start_time)

    print('Avg execution time (ms): {:.2f}'.format(np.mean(time_spent)*1000))
    print('FPS=',1/np.mean(time_spent))
    print(time_spent)



