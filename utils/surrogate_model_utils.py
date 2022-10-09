import torch
import torchvision
from classifier_models import Vgg16_all_layer, Vgg19_all_layer,Res152_all_layer, Dense169_all_layer
from classifier_models.res_models import resnet152
import sys

def get_predict(out):
    out = out.sigmoid()
    max_pred, _ = out.max(dim=1, keepdim=True)
    positive_thresh = max_pred * (1/2)
    predict = (out > positive_thresh)

    return predict
    
def surr_model_cls(args, num_classes):
    if args.model_type == 'vgg19':
        model = torchvision.models.vgg19(pretrained=True)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
        pretrained_path = 'classifier_models/{}_checkpoints/vgg19/best_model.pth'.format(args.data_name)
        model.load_state_dict(torch.load(pretrained_path))
    elif args.model_type == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
        pretrained_path = 'classifier_models/{}_checkpoints/vgg16/best_model.pth'.format(args.data_name)
        model.load_state_dict(torch.load(pretrained_path))
    elif args.model_type == 'res152':
        model = resnet152(pretrained=True, num_classes=num_classes, use_conv_fc=True)
        pretrained_path = 'classifier_models/{}_checkpoints/res152/best_model.pth'.format(args.data_name)
        model.load_state_dict(torch.load(pretrained_path))
    elif args.model_type == 'dense169':
        model = torchvision.models.densenet169(pretrained=True)
        model.classifier = torch.nn.Linear(1664, num_classes)
        pretrained_path = 'classifier_models/{}_checkpoints/dense169/best_model.pth'.format(args.data_name)
        model.load_state_dict(torch.load(pretrained_path))
    else:
        raise Exception('Please check the model_type: {}'.format(args.model_type))
    
    return model

def surr_model_feat(args, num_classes):
    if args.model_type == 'vgg16':
        model = Vgg16_all_layer.Vgg16(num_classes, args.data_name)
        layer_idx = 16  # Maxpooling.3
    elif args.model_type == 'vgg19':
        model = Vgg19_all_layer.Vgg19(num_classes, args.data_name)
        layer_idx = 18  # Maxpooling.3
    elif args.model_type == 'res152':
        model = Res152_all_layer.Resnet152(num_classes, args.data_name)
        layer_idx = 5   # Conv3_8
    elif args.model_type == 'dense169':
        model = Dense169_all_layer.Dense169(num_classes, args.data_name)
        layer_idx = 6  # Denseblock.2
    else:
        raise Exception('Please check the model_type: {}'.format(args.model_type))

    return model, layer_idx

# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
def clip_normalize(t):
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

    return t


def normalize(t):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t_tmp = t.clone()
    t_tmp[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t_tmp[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t_tmp[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

    return t_tmp

def denormalize(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)


def exists(val):
    return val is not None
    
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new