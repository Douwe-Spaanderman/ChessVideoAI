import os
import numpy as np
import torch
from PIL import Image
import glob

class ChessPiecesData(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned with the labels
        # Remove files in directory which aren't ending with png or txt (for image and boxes)
        self.imgs = glob.glob(root +'images/*.png')
        self.rect = glob.glob(root +'label/*.txt')    

    def __getitem__(self, idx):
        # load images and box
        img_path = self.imgs[idx]
        box_path = self.rect[idx]
        # open image and convert to rgb from bgr
        img = Image.open(img_path).convert("RGB")
        box = read_yolo(box_path, img)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(box[:,1:], dtype=torch.float32)
        labels = torch.as_tensor((box[:,0]), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (box[:, 4] - box[:, 2]) * (box[:, 3] - box[:, 1])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(box),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
    
        print(target)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        return len(self.imgs)