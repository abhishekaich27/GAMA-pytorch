import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import xml.etree.ElementTree as ET

import os


class VOCClsDataset(data.Dataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor')

    def __init__(self, root_dir, ann_file, img_dir, phase='train'):
        assert phase in ['train', 'test']
        self.phase = phase
        self.root_dir = root_dir
        ann_file = self._to_list(ann_file)
        img_dir = self._to_list(img_dir)
        self.anns = self._load_anns(ann_file, img_dir)

        self.cat2label = {v: k for k, v in enumerate(self.CLASSES)}
        self.label2cat = {v: k for k, v in self.cat2label.items()}
        #scale_size = 256
        #img_size = 224
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)), # for inception = 299, for resnet = 224
                transforms.ToTensor(),
                ])

    def __len__(self):
        return len(self.anns)

    def _to_list(self, element):
        if isinstance(element, list):
            return element
        else:
            return [element]

    def _load_anns(self, ann_file, img_dir):
        assert isinstance(ann_file, list)
        anns = []
        for file, imgdir in zip(ann_file, img_dir):
            with open(os.path.join(self.root_dir, file), 'r') as f:
                for line in f:
                    name = line.strip().split()[0]
                    anns.append((imgdir, name))
        return anns

    def _load_single_img(self, img_file):
        img = Image.open(img_file).convert('RGB')
        img = self.transform(img)
        return img

    def _get_ann_info(self, ann_file):
        tree = ET.parse(ann_file)
        root = tree.getroot()
        labels = []
        for obj in root.findall('object'):
            label = self.cat2label[obj.find('name').text]
            difficult = int(obj.find('difficult').text)
            if not difficult:
                labels.append(label)
        label = np.unique(np.sort(np.array(labels)))
        return label

    def __getitem__(self, idx):
        img_dir, name = self.anns[idx]
        ann_file = os.path.join(self.root_dir, img_dir, 
            'Annotations/{}.xml'.format(name))
        img_file = os.path.join(self.root_dir, img_dir,
            'JPEGImages/{}.jpg'.format(name))
        label = self._get_ann_info(ann_file)
        img = self._load_single_img(img_file)

        onehot = torch.zeros(len(self.CLASSES)).float()
        onehot[label] = 1
        onehot = onehot.float()

        if self.phase == 'train':
            return img, onehot
        else:
            return img, onehot


if __name__ == '__main__':
    dataset = VOCClsDataset(
        root_dir='/home1/share/pascal_voc/VOCdevkit/VOC2007_test',
        ann_file=[
            'VOC2007_test/ImageSets/Main/test.txt',],
            # 'VOC2012/ImageSets/Main/test.txt'],
        img_dir=['VOC2007_test', ],
        #'VOC2012'],
        phase='train')
    print(len(dataset))
    print(dataset[1000])