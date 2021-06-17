import os
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import glob
import random
import time
from ChessVideoAI.classes import ChessPiecesData

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def define_model():
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 13  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 13
    # use our dataset and defined transformations
    dataset = ChessPiecesData("./data/", get_transform(train=False))
    dataset_test = ChessPiecesData("./data/", get_transform(train=False))
    
    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn)

    # get the model using our helper function
    model = define_model()

    # move model to the right device
    model = model.to(device)

    # construct an optimizer
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=0.006)

    model.train()
    from tqdm import tqdm
    epoch = 2
    for epoch in tqdm(range(1)):
        for images,targets in tqdm(data_loader):
            images = list(image.to(device) for image in images)
            
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()

            optimizer.zero_grad()
            optimizer.step()
            
        print("Loss = {:.4f} ".format(losses.item()))

    torch.save(model.state_dict(), './model.pth')

main()