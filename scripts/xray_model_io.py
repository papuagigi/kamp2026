"""Shared image/label adapters for the controlled X-ray model comparison."""
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from xray_config import ROOT

PREPARED=ROOT/'data/xray_roadmap_20261002'


def seed_all():
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    torch.set_num_threads(6)


def faster_model(pretrained=True, resolution=512):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    model=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,
        min_size=resolution,max_size=resolution,box_score_thresh=.001,box_detections_per_img=50,
        rpn_pre_nms_top_n_train=200,rpn_post_nms_top_n_train=100,
        rpn_pre_nms_top_n_test=200,rpn_post_nms_top_n_test=100,box_batch_size_per_image=64)
    if pretrained:
        model.load_state_dict(torch.load(ROOT/'weights/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth',map_location='cpu',weights_only=True))
    model.roi_heads.box_predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,2)
    model.rpn.anchor_generator=AnchorGenerator(((8,),(16,),(32,),(64,),(128,)),((.5,1.,2.),)*5)
    return model


class XrayDataset(Dataset):
    def __init__(self, directory, split='train', limit=None):
        self.directory=Path(directory);self.split=split
        self.paths=sorted((self.directory/'images'/split).glob('*.png'))
        if limit:self.paths=self.paths[:limit]
        if not self.paths:raise ValueError(f'No images: {directory}/{split}')

    def __len__(self):return len(self.paths)

    def __getitem__(self,index):
        p=self.paths[index];im=Image.open(p).convert('RGB');w,h=im.size
        txt=self.directory/'labels'/self.split/f'{p.stem}.txt'
        values=np.array([list(map(float,l.split())) for l in txt.read_text().splitlines() if l.strip()]).reshape(-1,5)
        xyxy=np.column_stack([(values[:,1:3]-values[:,3:]/2)*[w,h],(values[:,1:3]+values[:,3:]/2)*[w,h]])
        target={'boxes':torch.tensor(xyxy,dtype=torch.float32),'labels':torch.ones(len(values),dtype=torch.int64),
                'image_id':torch.tensor(index),'area':torch.tensor(values[:,3]*w*values[:,4]*h,dtype=torch.float32),
                'iscrowd':torch.zeros(len(values),dtype=torch.int64)}
        return pil_to_tensor(im).float()/255,target


def collate(batch):return tuple(zip(*batch))


def sync(device):
    if str(device)=='mps':torch.mps.synchronize()
    elif str(device).startswith('cuda'):torch.cuda.synchronize()
