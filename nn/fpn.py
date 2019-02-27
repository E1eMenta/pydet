from torch import nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.top_layer = nn.Conv2d(in_channels_list[-1], out_channels, 1)

        in_channels_list = in_channels_list[:-1]

        self.lateral_layers = nn.ModuleList(
            [nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list]
        )

        self.smooth_layers = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1)]
        )

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y


    def forward(self, layers):

        last_layer = self.top_layer(layers[-1])
        out_layers = [last_layer]

        N = len(layers) - 2
        for layer_idx in range(N, -1, -1):
            lateral_layer = self.lateral_layers[layer_idx]
            lateral = lateral_layer(layers[layer_idx])

            last_layer = self._upsample_add(last_layer, lateral)
            out_layers.append(last_layer)

        out_layers = list(reversed(out_layers))
        return out_layers