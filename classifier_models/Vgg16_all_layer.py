import torchvision.models as models
import torch


class Vgg16(torch.nn.Module):
    def __init__(self, num_classes, data_name):
        super(Vgg16, self).__init__()
        
        model = models.vgg16(pretrained=True)
        
        if data_name in ['voc', 'coco']:
            model.classifier[6] = torch.nn.Linear(4096, num_classes)
            pretrained_path = 'classifier_models/{}_checkpoints/vgg16/best_model.pth'.format(data_name)
            model.load_state_dict(torch.load(pretrained_path))
            
        self.vgg = model.eval()
        self.features = list(self.vgg.features)
        self.model = torch.nn.ModuleList(self.features)

    def forward(self, x):
        layers = []
        for ii, model_ in enumerate(self.model):
            x = model_(x)
            layers.append(x.clone())

        x = self.vgg.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.vgg.classifier(x)
        layers.append(x.clone())
        return layers

if __name__ == "__main__":
    # pass
    layer = Vgg16(20, 'voc')(torch.zeros(2,3,224,224))
    print(len(layer))
