import torchvision.models as models
import torch


class Dense169(torch.nn.Module):
    def __init__(self, num_classes, data_name):
        super(Dense169, self).__init__()
        model = models.densenet169(pretrained=True)

        if data_name in ['voc', 'coco']:
            model.classifier = torch.nn.Linear(1664, num_classes)
            pretrained_path = 'classifier_models/{}_checkpoints/dense169/best_model.pth'.format(data_name)
            model.load_state_dict(torch.load(pretrained_path))

        self.model = model.eval()
        features = list(self.model.features)
        self.features = torch.nn.ModuleList(features)

    def forward(self, x):
        layers = []
        for ii, model_ in enumerate(self.features):
            x = model_(x.clone())
            layers.append(x.clone())
        return layers



if __name__ == "__main__":
    # pass
    layer = Dense169(20, 'voc')(torch.zeros(2,3,224,224))
    print(len(layer))

