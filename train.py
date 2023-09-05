import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from classifier_models import Vgg16_all_layer, Vgg19_all_layer, Res152_all_layer, Res50_all_layer, Dense121_all_layer, Dense169_all_layer
import random
from generator import GeneratorResnet
from ml_dataset import CocoClsDataset, VOCClsDataset
from utils.surrogate_model_utils import *
import pprint
import copy
from itertools import product
from torchvision import transforms
import clip
from eval import evaluate_ml
import pickle
from losses import ContrastiveLoss


parser = argparse.ArgumentParser(description='classifier attack')
parser.add_argument('--train_dir', default='', help='the path for training data') 
parser.add_argument('--batch_size', type=int, default=9, help='Number of trainig samples/batch')
parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for adam')
parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget')
parser.add_argument('--seed', type=int, default=42, help='Reproducibility seed')
parser.add_argument('--margin', type=float, default=0.2, help='Margin for contrastive loss')
parser.add_argument('--surr_model_type', type=str, default='vgg16',
                    help='Model against GAN is trained: vgg16, vgg19 res152, dense169')
parser.add_argument('--attack_model_type', type=str, default='vgg16',
                    help='Model to be attacked: vgg16, vgg19 res152, dense169')
parser.add_argument('--clip_backbone', type=str, default='ViT-B/16',
                    help='Clip Model to be loaded: [RN50, RN101, ViT-B/32, ViT-L/14, ViT-B/16]') #  RN50x16, RN50x4 require different image size
parser.add_argument('--data_name', default='voc', help='coco, voc')
parser.add_argument('--save_folder', type=str, help='Folder to save trained models')
parser.add_argument('--loss_type', type=str, default='contrastive', help='contrastive')

args = parser.parse_args()
pprint.pprint(args)

# Normalize (0-1)
eps = args.eps/255.0

# Fix seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(args.seed)

# GPU/CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up folder
if not os.path.isdir(args.save_folder):
    os.mkdir(args.save_folder)

trained_gen_path = '{}/{}'.format(args.save_folder, args.data_name)
if not os.path.isdir(trained_gen_path):
    os.mkdir(trained_gen_path)

trained_gen_path = trained_gen_path + '/{}'.format(args.surr_model_type)
if not os.path.isdir(trained_gen_path):
    os.mkdir(trained_gen_path)

# Get CLIP name for folder
clip_model_name = copy.deepcopy(args.clip_backbone)
if clip_model_name[:3] == 'ViT':
    clip_model_name = clip_model_name.replace("-", "").replace("/", "")

exp_name = '{}_{}_w_{}'.format(args.loss_type, args.surr_model_type, clip_model_name)
trained_gen_path = trained_gen_path + '/{}'.format(exp_name)

if not os.path.isdir(trained_gen_path):
    os.mkdir(trained_gen_path)

print('Models to be saved in path: ', trained_gen_path)

# Training/Testing Data
if args.data_name == 'coco':
    train_dataset = CocoClsDataset(root_dir=args.train_dir,
                                ann_file='annotations/instances_train2017.json',
                                img_dir='images/train2017',
                                phase='train')

    test_dataset = CocoClsDataset(root_dir='/data/AmitRoyChowdhury/msCOCO/',
                                ann_file='annotations/instances_val2017.json',
                                img_dir='images/val2017',
                                phase='test')
    num_classes = len(train_dataset.coco.dataset['categories'])
    class_list = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane',
                    'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter',
                    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                    'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite',
                    'baseball bat', 'baseball glove', 'skateboard', 'surfboard']
    
    # Dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size//2, shuffle=False, num_workers=2)

elif args.data_name == 'voc':
    train_dataset = VOCClsDataset(root_dir=args.train_dir,
                                ann_file=['VOC2007/ImageSets/Main/trainval.txt',
                                        'VOC2012/ImageSets/Main/trainval.txt'],
                                img_dir=['VOC2007', 'VOC2012'],
                                phase='train')

    test_dataset = VOCClsDataset(root_dir='/data/AmitRoyChowdhury/pascalVOC/VOCdevkit',
                                ann_file='VOC2007_test/ImageSets/Main/test.txt',
                                img_dir=['VOC2007_test'],
                                phase='test')
    num_classes = len(train_dataset.CLASSES)

    class_list = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor']

    # Dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size//2, shuffle=False, num_workers=2)
    

train_size = len(train_dataset)
print('Training data size:', train_size)

# Set up CLIP model
model_clip, _ = clip.load(args.clip_backbone) # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
model_clip = model_clip.to(device).eval()

# Set up adjacency matrix to generate plausible sentences (needed only for VOC and COCO)
adj_file_path = 'ml_data/{0}/{0}_adj.pkl'.format(args.data_name)
adj_dict = pickle.load(open(adj_file_path, 'rb'))
adj_mat = adj_dict['adj']
adj_mat[adj_mat>1] = 1.0

# Create pairs
class_list_pairs = list(product(enumerate(class_list), repeat=2))

# Only create pairs that exist in real world based on adj matrix
clist_pruned = [[[idx1, idx2], ["a photo depicts {}".format(' and '.join([arr1, arr2]))]] for ((idx1, arr1), (idx2,arr2)) in class_list_pairs if adj_mat[idx1,idx2]>0]
    

# Extract text features 
if clip_model_name[:4] == 'ViTB':
    clist_text_feats = torch.zeros((len(clist_pruned), 512))
elif clip_model_name[:4] == 'ViTL':
    clist_text_feats = torch.zeros((len(clist_pruned), 768))
elif clip_model_name == 'RN101':
    clist_text_feats = torch.zeros((len(clist_pruned), 512))
elif clip_model_name == 'RN50':
    clist_text_feats = torch.zeros((len(clist_pruned), 1024))

for idx, element in enumerate(clist_pruned):
    text_tokens = clip.tokenize(element[1]).to(device)
    clist_text_feats[idx, :] = model_clip.encode_text(text_tokens).float().detach()
print('Intialized text feat matrix with size: ', clist_text_feats.shape)


# Surrogate Model (all last layers give 512)
if args.surr_model_type == 'vgg16':
    model = Vgg16_all_layer.Vgg16(num_classes, args.data_name)
    layer_idx = [15, 16, 17, 18] 
    layer_bia = 16
elif args.surr_model_type == 'vgg19':
    model = Vgg19_all_layer.Vgg19(num_classes, args.data_name)
    layer_idx = [15, 16, 17, 18, 19]  
    layer_bia = 18
elif args.surr_model_type == 'res152':
    model = Res152_all_layer.Resnet152(num_classes, args.data_name)
    layer_idx = [3, 4, 5] 
    layer_bia = 5
    if clip_model_name == 'RN50':
        layer_idx.append(6)
elif args.surr_model_type == 'res50':
    model = Res50_all_layer.Resnet50(num_classes, args.data_name)
    layer_idx = [3, 4, 5] 
    layer_bia = 5
elif args.surr_model_type == 'dense169':
    model = Dense169_all_layer.Dense169(num_classes, args.data_name)
    layer_idx = [5, 6]
    layer_bia = 6
elif args.surr_model_type == 'dense121':
    model = Dense121_all_layer.Dense121(num_classes, args.data_name)
    layer_idx = [5, 6]
    layer_bia = 6
else:
    raise Exception('Please check the surr_model_type')

model = model.eval()

# Generator
netG = GeneratorResnet()

# Loss for text features
criteria = ContrastiveLoss(args.margin)


# Loss for img features
criteriaMSE = nn.MSELoss().to(device)

# Data Parallel all the models
netG = nn.DataParallel(netG).to(device)
model = nn.DataParallel(model).to(device)

# Optimizer
optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

print('size of original text features:', clist_text_feats.shape)


# Training
with torch.autograd.set_detect_anomaly(True):
    tmp_score = 0
    for epoch in range(args.epochs):
        
        running_loss, counter = 0, 0

        print("*----------- Epoch: {}. Training with eps: {}.".format(epoch, int(eps*255.0)))

        tk0 = tqdm(train_dataloader, total=int(len(train_dataloader)))

        for i, (img, _) in enumerate(tk0):
            
            img = img.to(device)
            optimG.zero_grad()
            
            # perturbed image
            adv = netG(img)
            
            # projection
            adv = torch.min(torch.max(adv, img - eps), img + eps)
            adv = torch.clamp(adv, 0.0, 1.0)
            
            # compute loss
            img_feats = model(normalize(img))[layer_idx[-1]] # 512 dim size feats
            adv_feats = model(normalize(adv))[layer_idx[-1]] # 512 dim size feats

            adv_feats = torch.mean(torch.mean(adv_feats, -1), -1) # 512 dim size embeddings
            img_feats = torch.mean(torch.mean(img_feats, -1), -1) # 512 dim size embeddings
            
            # randomly sample text features from pre-computed text feats
            text_feat_idx = random.sample(range(1, clist_text_feats.shape[0]), img_feats.shape[0])
            
            batch_clist_text_feats = clist_text_feats[text_feat_idx, :].to(device)
            
            # find the least similar indices in the current img batch
            clip_img_feats = model_clip.encode_image(clip_normalize(img)).float()
            clip_img_feats = clip_img_feats/clip_img_feats.norm(dim=-1, keepdim=True)
            
            batch_clist_text_feats_tmp = batch_clist_text_feats/batch_clist_text_feats.norm(dim=-1, keepdim=True)
            sim = np.absolute(batch_clist_text_feats_tmp.detach().cpu().numpy() @ clip_img_feats.detach().cpu().numpy().T)
            indices = np.argmin(sim, axis=0)

            # select the least similar text feats for every img
            batch_clist_text_feats = batch_clist_text_feats[indices, :]
            
            # compute loss w.r.t. CLIP text feats
            # anchor, negatives, positives = adv_feats, img_feats, batch_clist_text_feats 
            loss = criteria(adv_feats, img_feats, batch_clist_text_feats)
            
            # compute loss w.r.t. CLIP img feats
            adv_feats = adv_feats/adv_feats.norm(dim=-1, keepdim=True)
            
            loss -= criteriaMSE(adv_feats, clip_img_feats)

            # BIA tuning 
            img_feats_tensor = model(normalize(img))[layer_bia]
            adv_feats_tensor = model(normalize(adv))[layer_bia] 
            attention = torch.mean(img_feats_tensor, dim=1, keepdim=True).detach()
        
            loss += torch.cosine_similarity((adv_feats_tensor*attention).reshape(adv_feats_tensor.shape[0], -1), 
                            (img_feats_tensor*attention).reshape(img_feats_tensor.shape[0], -1)).mean()
            
            # backpropagate
            loss = loss/3
            loss.backward()
            optimG.step()

            counter += 1
            
            running_loss += loss.item()
            tk0.set_postfix(loss = (running_loss/counter), epoch = epoch)
            tk0.refresh()
        tk0.close()
        

        # evaluate 
        if epoch == (args.epochs-1):
            evaluate_ml(args, eps, num_classes, test_dataloader, netG.module.state_dict(), device)
            model_path = os.path.join(trained_gen_path, 'netG_{}.pth'.format(epoch))
            torch.save(netG.module.state_dict(), model_path)
    

