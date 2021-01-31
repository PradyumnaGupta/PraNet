import argparse

my_parser = argparse.ArgumentParser()

my_parser.add_argument('--base_path')
my_parser.add_argument('--output_path')

args = my_parser.parse_args()

# !pip install segmentation-models-pytorch
# !pip install -U git+https://github.com/albu/albumentations --no-cache-dir

x_train_dir = args.base_path+"/train/clean_patch/x/"
y_train_dir = args.base_path+"/train/clean_patch/gt/"
x_val_dir = args.base_path+"/val/clean_patch/x/"
y_val_dir = args.base_path+"/val/clean_patch/gt/"
# x_test_dir = "/content/drive/MyDrive/SegPC-ISBI/New/val/clean_patch/x/"
# # y_test_dir = "/content/drive/MyDrive/SegPC-ISBI/New/val/clean_patch/gt/"

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import torch
from torch import nn
import numpy as np
import segmentation_models_pytorch as smp
import thop
import re
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 

import sys 
sys.path.append(args.base_path+"/PraNet/")
from PraNet.lib.PraNet_Res2Net import PraNet
from PraNet.MyTrain import structure_loss


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    
seed_everything(42)   


class Dataset(BaseDataset):
    """ Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['background', 'cytoplasm', 'nucleus']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
      
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = np.array(Image.open(self.images_fps[i]))
        # mask = plt.imread(self.masks_fps[i], "bmp")
        mask = cv2.imread(self.masks_fps[i],0)
        
        mask[mask==33] = 0
        mask[mask==151] = 1
        mask[mask==175] = 2
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask
        
    def __len__(self):
        return len(self.ids)

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        # plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def get_training_augmentation():
    train_transform = [

        # albu.HorizontalFlip(p=0.5),

        # albu.ShiftScaleRotate(p=0.75),

        albu.PadIfNeeded(min_height=384, min_width=512, always_apply=True),
        # albu.RandomCrop(height=320, width=320, always_apply=True),

        # albu.IAAAdditiveGaussianNoise(p=0.1),
        # albu.IAAPerspective(p=0.1),

        # albu.OneOf(
        #     [
        #         albu.CLAHE(p=0.1),
        #         albu.RandomBrightness(p=0.1),
        #         albu.RandomGamma(p=0.1),
        #     ],
        #     p=0.5,
        # ),

        # albu.OneOf(
        #     [
        #         albu.IAASharpen(p=0.1),
        #         albu.Blur(blur_limit=3, p=0.1),
        #         albu.MotionBlur(blur_limit=3, p=0.1),
        #     ],
        #     p=0.5,
        # ),
        # albu.OneOf(
        #     [
        #         albu.RandomContrast(p=0.1),
        #         albu.HueSaturationValue(p=0.1),
        #     ],
        #     p=0.5,
        # )
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 512)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

CLASSES = ["background", "cytoplasm", 'nucleus']
if torch.cuda.is_available():
    DEVICE = 'cuda'
else :
    DEVICE = 'cpu'    

preprocessing_fn = None


train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_val_dir, 
    y_val_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=1)

models = [PraNet() for i in range(3)]

class BaseObject(nn.Module):

    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name


class Loss(BaseObject):

    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError('Loss should be inherited from `Loss` class')

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError('Loss should be inherited from `BaseLoss` class')

    def __rmul__(self, other):
        return self.__mul__(other)


class SumOfLosses(Loss):

    def __init__(self, l1, l2):
        name = '{} + {}'.format(l1.__name__, l2.__name__)
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2

    def __call__(self, *inputs):
        return self.l1.forward(*inputs) + self.l2.forward(*inputs)


class MultipliedLoss(Loss):

    def __init__(self, loss, multiplier):

        # resolve name
        if len(loss.__name__.split('+')) > 1:
            name = '{} * ({})'.format(multiplier, loss.__name__)
        else:
            name = '{} * {}'.format(multiplier, loss.__name__)
        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def __call__(self, *inputs):
        return self.multiplier * self.loss.forward(*inputs)

class EMLoss(Loss):
  def __init__(self):
    super(EMLoss,self).__init__()

  def forward(self,inputs,targets):
    # inputs = torch.nn.Sigmoid()(inputs)
    return structure_loss(inputs,targets)  


losses = [[EMLoss() for i in range(4)] for i in range(3)]
metric = smp.utils.metrics.IoU(threshold=0.5)
optimizers = [torch.optim.Adam([ 
    dict(params=model.parameters(), lr=1e-4),
]) for model in models]
lrs = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[i], T_max = 10) for i in range(3)]

class AverageMeter:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0
    
    def update(self,val,n=1):
        self.val=val
        self.sum = self.sum + val*n
        self.count = self.count + n
        self.avg=self.sum/self.count

from utils.utils import clip_gradient, adjust_lr
size_rates = [0.75, 1, 1.25]

def model_update(model,optimizer,criterions,x,y):
  x= x.to(DEVICE)
  y= y.to(DEVICE)

  for rate in size_rates:  
    optimizer.zero_grad()

    # trainsize = int(round(2624*rate/32)*32)
    if rate != 1:
        images = F.upsample(x, scale_factor=rate, mode='bilinear', align_corners=True)
        gts = F.upsample(y.unsqueeze(1), scale_factor=rate, mode='bilinear', align_corners=True)

    else :    
      images = x
      gts = y.unsqueeze(1)

    predictions = model.forward(images)
    loss = sum([criterion(predictions[i],gts) for criterion in criterions])
    loss.backward()
    clip_gradient(optimizer, 0.5)
    optimizer.step()

    if rate==1:
      prediction_to_return = predictions[-1]
      loss_to_return = torch.mean(loss)

  return prediction_to_return,loss_to_return

def model_evaluate(model,criterions,x,y):
  x = x.to(DEVICE)
  y = y.to(DEVICE)  
  
  with torch.no_grad():
    predictions = model.forward(x)
    loss = sum([criterion(torch.nn.Sigmoid()(predictions[i]),y.unsqueeze(1)) for criterion in criterions])

  return predictions[-1],torch.mean(loss)

loss_meters = [AverageMeter() for i in range(3)]
metric_meter = AverageMeter()
metric_meter_activated = AverageMeter()
metric_meter_activated2 = AverageMeter()

models = [model.to(DEVICE) for model in models]

max_score = 0
path = ''

if not os.path.exists(args.base_path+"/logs"):
    os.makedirs(args.base_path+"/logs")

f=open("pranet_logs.txt","w")

for epoch in range(100):
  print('\nEpoch: {}'.format(epoch))
  f.write('\nEpoch: {} \n'.format(epoch))
  # Training 

  metric_meter_activated.reset()
  metric_meter_activated2.reset()
  for i in range(len(loss_meters)) : loss_meters[i].reset()

  progress = tqdm(train_loader,total=len(train_loader))

  models = [model.train() for model in models]
  [adjust_lr(optimizers[i],1e-4,epoch,0.1,50) for i in range(3)]

  for step,(x,y) in enumerate(progress):

    outputs = []

    for i in range(3):

      y_pred,loss_value = model_update(models[i],optimizers[i],losses[i],x,y[:,i,:,:])
      loss_meters[i].update(loss_value)
      outputs.append(y_pred)

    y_pred = torch.stack(outputs,dim=1).squeeze()

    if len(y_pred.shape)!=4: y_pred = y_pred.unsqueeze(0)

    metric_value_activated = metric(torch.nn.Softmax2d()(y_pred),y.to(DEVICE)).cpu().detach()
    metric_meter_activated.update(metric_value_activated)

    metric_value_activated2 = metric(torch.nn.Sigmoid()(y_pred),y.to(DEVICE)).cpu().detach()
    metric_meter_activated2.update(metric_value_activated2)

    progress.set_postfix(bloss=loss_meters[0].avg.item(),closs=loss_meters[1].avg.item(),nloss=loss_meters[2].avg.item(),iou_score_activated=metric_meter_activated.avg.item(),iou_score_activated2=metric_meter_activated2.avg.item())
    break
  f.write("bloss:{} closs:{} nloss:{} iou_score:{} \n".format(loss_meters[0].avg.item(),loss_meters[1].avg.item(),loss_meters[2].avg.item(),metric_meter_activated.avg.item()))

  for i in range(3): lrs[i].step()

  # Validation
  
  metric_meter_activated.reset()
  metric_meter_activated2.reset()
  for i in range(len(loss_meters)) : loss_meters[i].reset()

  progress = tqdm(valid_loader,total=len(valid_loader))

  models = [model.eval() for model in models]

  for step,(x,y) in enumerate(progress):

    outputs = []

    for i in range(3):

      y_pred,loss_value = model_evaluate(models[i],losses[i],x,y[:,i,:,:])
      loss_meters[i].update(loss_value)
      outputs.append(y_pred)

    y_pred = torch.stack(outputs,dim=1).squeeze()

    if len(y_pred.shape)!=4: y_pred = y_pred.unsqueeze(0)

    metric_value_activated = metric(torch.nn.Softmax2d()(y_pred),y.to(DEVICE)).cpu().detach()
    metric_meter_activated.update(metric_value_activated)

    metric_value_activated2 = metric(torch.nn.Sigmoid()(y_pred),y.to(DEVICE)).cpu().detach()
    metric_meter_activated2.update(metric_value_activated2)

    progress.set_postfix(bloss=loss_meters[0].avg.item(),closs=loss_meters[1].avg.item(),nloss=loss_meters[2].avg.item(),iou_score_activated=metric_meter_activated.avg.item(),iou_score_activated2=metric_meter_activated2.avg.item())
    break
  f.write("bloss:{} closs:{} nloss:{} iou_score:{} \n".format(loss_meters[0].avg.item(),loss_meters[1].avg.item(),loss_meters[2].avg.item(),metric_meter_activated.avg.item()))
  f.write("\n")

  if metric_meter_activated.avg > max_score:
    if not os.path.exists(args.output_path+"/model_saves"):
        os.makedirs(args.output_path+"/model_saves")  
    max_score=metric_meter_activated.avg
    if len(path):
        os.system("rm -r {}".format(path))
    path = args.output_path+'/model_saves/TRIPLE_PRANET-{}.pth'.format(max_score)
    torch.save({
        'optimizers':optimizers,
        'models':models
    },path)
    print("model saved !!")
  break

f.close()