import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models.base_backbone_entropy import FewShotModel


class ProtoNet(FewShotModel):
    def __init__(self, args):
        super().__init__(args)

        self.temp = nn.Parameter(torch.tensor(10., requires_grad=True))
        self.method = 'cos'

    def compute_logits(self, feat, proto, metric='dot', temp=1.0):
        assert feat.dim() == proto.dim()

        if feat.dim() == 2:
            if metric == 'dot':
                logits = torch.mm(feat, proto.t())
            elif metric == 'cos':
                logits = 1 - torch.mm(F.normalize(feat, dim=-1),
                                      F.normalize(proto, dim=-1).t())
            elif metric == 'sqr':
                logits = -(feat.unsqueeze(1) -
                           proto.unsqueeze(0)).pow(2).sum(dim=-1)

        elif feat.dim() == 3:
            if metric == 'dot':
                logits = torch.bmm(feat, proto.permute(0, 2, 1))
            elif metric == 'cos':
                logits = torch.bmm(F.normalize(feat, dim=-1),
                                   F.normalize(proto, dim=-1).permute(0, 2, 1))
            elif metric == 'sqr':
                logits = -(feat.unsqueeze(2) -
                           proto.unsqueeze(1)).pow(2).sum(dim=-1)
        return logits * temp

    def _forward(self, x_shot, x_query):

        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)
            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        logits = self.compute_logits(
            x_query, x_shot, metric=metric, temp=self.temp)

        logits = logits.view(-1, self.args.way)

        return logits