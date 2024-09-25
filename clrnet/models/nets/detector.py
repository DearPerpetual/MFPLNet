import torch.nn as nn
import torch

from clrnet.models.registry import NETS
from ..registry import build_backbones, build_aggregator, build_heads, build_necks
from clrnet.models.backbones.swin_transformer import swin_t



@NETS.register_module
class Detector(nn.Module):
    def __init__(self, cfg):
        super(Detector, self).__init__()
        self.cfg = cfg
        self.backbone = swin_t()
        self.aggregator = build_aggregator(cfg) if cfg.haskey('aggregator') else None
        self.neck = build_necks(cfg) if cfg.haskey('neck') else None
        self.heads = build_heads(cfg)
    
    def get_lanes(self):
        return self.heads.get_lanes(output)

    def forward(self, batch):
        output = {}
        fea = self.backbone(batch['img'])

        if self.aggregator:
            fea[-1] = self.aggregator(fea[-1])

        if self.neck:
            fea = self.neck(fea)

        if self.training:
            output = self.heads(fea, batch=batch)
        else:
            output = self.heads(fea)

        return output

