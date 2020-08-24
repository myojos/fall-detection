import torch
from torch import nn
from .mobilenet import mobilenetv3


class TwoStreamFallNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.rgb_mobile = mobilenetv3(pretrained=True)
        self.flow_mobile = mobilenetv3(pretrained=True)
        self.__modify_flow_net_entry()
        self.__modify_nets_classifier()

    def __modify_flow_net_entry(self):
        first_conv = self.flow_mobile.features[0][0]
        average_rgb_conv = torch.mean(first_conv.weight, dim=1)
        flow_conv = torch.stack([average_rgb_conv for _ in range(20)]).permute(1, 0, 2, 3)
        self.flow_mobile.features[0][0] = flow_conv

    def __modify_nets_classifier(self):
        for net in [self.rgb_mobile, self.flow_mobile]:
            net.classifier[1] = nn.Linear(1280, 101)
            net.classifier.add_module('softmax', nn.Softmax(dim=-1))

    def forward(self, rgb, flow):
        """
        rgb:  N x 3 x H x W
        flow: N x (2L) x H x W
        """
        rgb_softmax = self.rgb_mobile(rgb)
        flow_softmax = self.flow_mobile(flow)
        return (rgb_softmax + 2 * flow_softmax) / 3
