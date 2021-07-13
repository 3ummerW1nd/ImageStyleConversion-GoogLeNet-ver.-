import torch
import torch.nn as nn

from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class google(nn.Module):
    def __init__(self):
        super(google, self).__init__()
        self.model = models.googlenet(pretrained=True)
        # print(self.model)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        outs = []
        x = self.model.conv1(x)
        outs.append(x)
        x = self.model.maxpool1(x)
        outs.append(x)
        x = self.model.conv2(x)
        outs.append(x)
        x = self.model.conv3(x)
        outs.append(x)
        x = self.model.maxpool2(x)
        outs.append(x)
        x = self.model.inception3a(x)
        outs.append(x)
        x = self.model.inception3b(x)
        outs.append(x)
        x = self.model.maxpool3(x)
        outs.append(x)
        x = self.model.inception4a(x)
        outs.append(x)
        x = self.model.inception4b(x)
        outs.append(x)
        x = self.model.inception4c(x)
        outs.append(x)
        x = self.model.inception4d(x)
        outs.append(x)
        x = self.model.inception4e(x)
        outs.append(x)
        x = self.model.maxpool4(x)
        outs.append(x)
        x = self.model.inception5a(x)
        outs.append(x)
        x = self.model.inception5b(x)
        outs.append(x)
        x = self.model.avgpool(x)
        outs.append(x)
        x = self.model.dropout(x)
        outs.append(x)
        return outs