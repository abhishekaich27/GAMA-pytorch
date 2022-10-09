import torchvision.models as models
import torch
from .res_models import resnet152


class Resnet152(torch.nn.Module):
    def __init__(self, num_classes, data_name):
        super(Resnet152, self).__init__()
        
        model = resnet152(pretrained=True, num_classes=num_classes, use_conv_fc=True)
        
        if data_name in ['voc', 'coco']:
            pretrained_path = 'classifier_models/{}_checkpoints/res152/best_model.pth'.format(data_name)
            model.load_state_dict(torch.load(pretrained_path))

        self.model = model.eval()
        features = list(self.model.children())
        self.features = torch.nn.ModuleList(features)
        self.internal = [ii for ii in range(9)]

    def forward(self, x):
        # pred = self.model(x)
        layers = []
        for ii, model_ in enumerate(self.features):
            if ii in self.internal:
                x = model_(x.clone())
                layers.append(x.clone())
        return layers

if __name__ == "__main__":
    # pass
    layer = Resnet152(20, 'voc')(torch.zeros(2,3,224,224))
    print(len(layer))



