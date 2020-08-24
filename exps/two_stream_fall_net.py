import torch
from torch import nn
from torchvision import models


class TwoStreamFallNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.rgb_mobile = models.mobilenet_v2(pretrained=True)
        self.flow_mobile = models.mobilenet_v2(pretrained=True)
        self.__modify_flow_net_entry()
        self.__modify_nets_classifier()

    def __modify_flow_net_entry(self):
        first_conv = self.rgb_mobile.features[0][0]
        average_rgb_conv_weight = torch.mean(first_conv.weight, dim=1)
        flow_conv_weight = average_rgb_conv_weight.unsqueeze(1).repeat(1, 20, 1, 1)
        new_conv = nn.Conv2d(20, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        new_conv.weight = nn.Parameter(flow_conv_weight)
        self.flow_mobile.features[0][0] = new_conv

    def __modify_nets_classifier(self):
        for net in [self.rgb_mobile, self.flow_mobile]:
            net.classifier[1] = nn.Linear(1280, 100)
            net.classifier.add_module('softmax', nn.Softmax(dim=-1))

    def forward(self, rgb, flow):
        """
        rgb:  N x  3 x H x W
        flow: N x 20 x H x W
        """
        rgb, flow = rgb.float(), flow.float()
        rgb_softmax = self.rgb_mobile(rgb)
        flow_softmax = self.flow_mobile(flow)
        return torch.log((rgb_softmax + 2 * flow_softmax) / 3)
