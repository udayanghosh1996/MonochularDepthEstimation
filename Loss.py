import torch
import torch.nn as nn
import torch.nn.functional as F


class MonoDepthLoss(nn.Module):
    def __init__(self, smoothness_weight=0.1):
        super(MonoDepthLoss, self).__init__()
        self.smoothness_weight = smoothness_weight

    def forward(self, output_depth, depth_map):
        if output_depth.shape[2:] != depth_map.shape[2:]:
            output_depth = F.interpolate(output_depth, size=depth_map.shape[2:], mode='bilinear', align_corners=False)

        l1_loss = torch.abs(output_depth - depth_map).mean()

        smoothness_loss = self.smoothness_loss(output_depth)

        total_loss = l1_loss + self.smoothness_weight * smoothness_loss

        return total_loss

    def smoothness_loss(self, depth_map):
        dzdx = torch.abs(depth_map[:, :, :, :-1] - depth_map[:, :, :, 1:])
        dzdy = torch.abs(depth_map[:, :, :-1, :] - depth_map[:, :, 1:, :])

        smoothness_loss = torch.sum(dzdx) + torch.sum(dzdy)

        return smoothness_loss
