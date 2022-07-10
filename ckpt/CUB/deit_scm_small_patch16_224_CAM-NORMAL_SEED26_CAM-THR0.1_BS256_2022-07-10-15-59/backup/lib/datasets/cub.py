import os
from PIL import Image
from pyparsing import original_text_for
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

def get_transforms(cfg):
    
    train_transform = transforms.Compose([
        transforms.Resize((cfg.DATA.RESIZE_SIZE, cfg.DATA.RESIZE_SIZE)),
        transforms.RandomCrop((cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
         transforms.Normalize(list(map(float, cfg.DATA.IMAGE_MEAN)), list(map(float, cfg.DATA.IMAGE_STD)))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(list(map(float, cfg.DATA.IMAGE_MEAN)), list(map(float, cfg.DATA.IMAGE_STD)))
    ])

    orig_transform = transforms.Compose([
        transforms.Resize((cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE)),
        transforms.ToTensor(),
        # transforms.Normalize(list(map(float, cfg.DATA.IMAGE_MEAN)), list(map(float, cfg.DATA.IMAGE_STD)))
    ])
    
    test_tencrops_transform = transforms.Compose([
        transforms.Resize((cfg.DATA.RESIZE_SIZE, cfg.DATA.RESIZE_SIZE)),
        transforms.TenCrop(cfg.DATA.CROP_SIZE),
        transforms.Lambda(lambda crops: torch.stack(
                [transforms.Normalize(cfg.DATA.IMAGE_MEAN, cfg.DATA.IMAGE_STD)
                 (transforms.ToTensor()(crop)) for crop in crops])),
    ])
    return train_transform, test_transform, test_tencrops_transform, orig_transform


class CUBDataset(Dataset):
    """ 'CUB <http://www.vision.caltech.edu/visipedia/CUB-200.html>'

    Args:
        root (string): Root directory of dataset where directory "CUB_200_2011" exists.
        cfg (dict): Hyperparameter configuration.
        is_train (bool): If True. create dataset from training set, otherwise creates from test set.
        val (bool): validation dataset for finetuning hyperparameters.
    """
    def __init__(self, root, cfg, is_train, val=False):

        self.root = root
        self.cfg = cfg
        self.is_train = is_train
        self.resize_size = cfg.DATA.RESIZE_SIZE
        self.crop_size = cfg.DATA.CROP_SIZE

        with open(os.path.join(root, 'images.txt'), 'r') as o:
            self.image_list = self.remove_1st_column(o.readlines())
        with open(os.path.join(root, 'image_class_labels.txt'), 'r') as o:
            self.label_list = self.remove_1st_column(o.readlines())
        with open(os.path.join(root, 'train_test_split.txt'), 'r') as o:
            self.split_list = self.remove_1st_column(o.readlines())
        with open(os.path.join(root, 'bounding_boxes.txt'), 'r') as o:
            self.bbox_list = self.remove_1st_column(o.readlines())
            
        self.train_transform, self.onecrop_transform, self.tencrops_transform, self.orig_transform = get_transforms(cfg)
        if cfg.TEST.TEN_CROPS:
            self.test_transform = self.tencrops_transform
        else:
            self.test_transform = self.onecrop_transform

        if is_train:
            self.index_list = self.get_index(self.split_list, '1')
        else:
            self.index_list = self.get_index(self.split_list, '0')
        
        self.val = val
        if val:
            self.image_dir = os.path.join(self.root, 'CUBV2')
            # val2/1/1.jpeg,1
            datalist = os.path.join(self.root, 'CUBV2', 'val', 'image_ids.txt')
            labelList = os.path.join(self.root, 'CUBV2', 'val', 'class_labels.txt')
            bboxlist = os.path.join(self.root, 'CUBV2', 'val', 'localization.txt')
            class_labels = {}
            boxes = {}
            dataList = []
            with open(datalist) as f:
                    for line in f.readlines():
                        dataList.append(line.strip('\n'))            
            with open(labelList) as f:
                for line in f.readlines():
                    image_id, class_label = line.strip('\n').split(',')
                    class_labels[image_id] = int(class_label)
            with open(bboxlist) as f:
                for line in f.readlines():
                    image_id, x0s, x1s, y0s, y1s = line.strip('\n').split(',')
                    x0, x1, y0, y1 = int(x0s), int(x1s), int(y0s), int(y1s)
                    if image_id in boxes:
                        boxes[image_id].append([x0, x1, y0, y1])
                    else:
                        boxes[image_id] = [[x0, x1, y0, y1]]                    
            
            self.val2_class_labels = class_labels
            self.val2_boxes = boxes
            self.val2_names = dataList

    def get_index(self, list, value):
        index = []
        for i in range(len(list)):
            if list[i] == value:
                index.append(i)
        return index

    def remove_1st_column(self, input_list):
        output_list = []
        for i in range(len(input_list)):
            if len(input_list[i][:-1].split(' '))==2:
                output_list.append(input_list[i][:-1].split(' ')[1])
            else:
                output_list.append(input_list[i][:-1].split(' ')[1:])
        return output_list

    def __getitem__(self, idx):
        name = self.image_list[self.index_list[idx]]
        image_path = os.path.join(self.root, 'images', name)
        
        label = int(self.label_list[self.index_list[idx]])-1
        
        if self.val:
            name = self.val2_names[idx]
            label = self.val2_class_labels[name]
            image = Image.open(os.path.join(self.image_dir, name)).convert('RGB')
            bbox = self.val2_boxes[name][0] # only one is available
        else:           
            image = Image.open(image_path).convert('RGB')
            bbox = self.bbox_list[self.index_list[idx]]
            bbox = [int(float(value)) for value in bbox]
            
        image_size = list(image.size)    
        
        if self.is_train:
            image = self.train_transform(image)
            return image, label
        else:
            orig = self.orig_transform(image)
            image = self.test_transform(image)

            [x, y, bbox_width, bbox_height] = bbox
            # if self.is_train:
            #     resize_size = self.resize_size
            #     crop_size = self.crop_size
            #     shift_size = (resize_size - crop_size) // 2
            resize_size = self.crop_size
            crop_size = self.crop_size
            shift_size = 0
            [image_width, image_height] = image_size
            left_bottom_x = int(max(x / image_width * resize_size - shift_size, 0))
            left_bottom_y = int(max(y / image_height * resize_size - shift_size, 0))
            right_top_x = int(min((x + bbox_width) / image_width * resize_size - shift_size, crop_size - 1))
            right_top_y = int(min((y + bbox_height) / image_height * resize_size - shift_size, crop_size - 1))

            # gt_bbox = [left_bottom_x, left_bottom_y, right_top_x, right_top_y]
            # gt_bbox = torch.tensor(gt_bbox)
            gt_bbox = np.array([left_bottom_x, left_bottom_y, right_top_x, right_top_y]).reshape(-1)
            gt_bbox = " ".join(list(map(str, gt_bbox)))
            
            return image, label, gt_bbox, name, orig

    def __len__(self):
        if self.val:
            return len(self.val2_names)
        else:
            return len(self.index_list)








