
import os
import random
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from torch.utils import data
import torchvision.transforms as transforms


#data augumentation
def cv_random_flip(img, label, depth, edge):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        edge = depth.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, depth, edge

def randomCrop(image, label, depth, edge):
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), depth.crop(random_region), edge.crop(random_region)

def randomRotation(image, label, depth, edge):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
        edge = depth.rotate(random_angle, mode)
    return image, label, depth, edge

def colorEnhance(image):
    #亮度
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    #对比度
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    #色度
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    #锐度
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


class train_dataset(data.Dataset):
    def __init__(self, root, image_size= 256):
        super(train_dataset, self).__init__()
        self.root = root
        self.image_size = [image_size, image_size]
        img_root = os.path.join(self.root, 'train_images')
        mask_root = os.path.join(self.root, 'train_masks')
        depth_root = os.path.join(self.root, 'train_depth')
        edge_root = os.path.join(self.root, 'train_edges')

        self.img_transform = transforms.Compose([
            transforms.Resize((self.image_size[0], self.image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.image_size[0], self.image_size[1])),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.image_size[0], self.image_size[1])),
            transforms.ToTensor()])
        self.edges_transform = transforms.Compose([
            transforms.Resize((self.image_size[0], self.image_size[1])),
            transforms.ToTensor()])

        file_names = os.listdir(img_root)
        self.img_names = []
        self.mask_names = []
        self.edge_names = []
        self.depth_names = []
        #读取数据集
        for i, name in enumerate(file_names):
            #只读取jpg结尾的文件
            if not name.endswith('.jpg'):
                continue
            self.mask_names.append(
                os.path.join(mask_root, name[:-4] + '.png')
            )
            self.img_names.append(
                os.path.join(img_root, name)
            )
            self.edge_names.append(
                os.path.join(edge_root, name[:-4] + '.png')
            )
            self.depth_names.append(
                os.path.join(depth_root, name[:-4] + '.png')
            )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        image = self.rgb_loader(self.img_names[index])
        mask = self.binary_loader(self.mask_names[index])
        depth = self.binary_loader(self.depth_names[index])
        edge = self.binary_loader(self.edge_names[index])

        # data augment
        image, mask, depth, edge = cv_random_flip(image, mask, depth, edge)
        image, mask, depth, edge = randomCrop(image, mask, depth, edge)
        image, mask, depth, edge = randomRotation(image, mask, depth, edge)
        image = colorEnhance(image)

        image = self.img_transform(image)
        mask = self.gt_transform(mask)
        depth = self.depths_transform(depth)
        edge = self.edges_transform(edge)
        return image, mask, depth, edge

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')



class test_dataset(data.Dataset):
    def __init__(self, root, image_size = 256):
        super(test_dataset, self).__init__()
        self.root = root
        self.image_size = [image_size, image_size]

        img_root = os.path.join(self.root, 'test_images')
        depth_root = os.path.join(self.root, 'test_depth')
        file_names = os.listdir(img_root)
        self.img_names = []
        self.names = []
        self.depth_names = []

        self.img_transform = transforms.Compose([
            transforms.Resize((self.image_size[0], self.image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.image_size[0], self.image_size[1])),
            transforms.ToTensor()])

        for i, name in enumerate(file_names):
            if not name.endswith('.jpg'):
                continue
            self.img_names.append(
                os.path.join(img_root, name)
            )
            self.names.append(name[:-4])
            self.depth_names.append(os.path.join(depth_root, name[:-4] + '.png'))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        image = self.rgb_loader(self.img_names[index])
        img_size = image.size
        img = self.img_transform(image)

        # load depth
        depth = self.binary_loader(self.depth_names[index])
        depth = self.depths_transform(depth)

        return img, depth, self.names[index], img_size

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
