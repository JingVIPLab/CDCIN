import torch
import torch.nn as nn
import numpy as np
from model.networks.backbone_entropy import VQABackbone

class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.encoder = VQABackbone(args)

    # test: [[[5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19]]]
    def split_instances(self, data):
        args = self.args
        if self.training:
            return (torch.Tensor(np.arange(args.way * args.shot)).long().view(1, args.shot, args.way),
                    torch.Tensor(np.arange(args.way * args.shot, args.way * (args.shot + args.query))).long().view(1,
                                                                                                                   args.query,
                                                                                                                   args.way))
        else:
            return (
                torch.Tensor(np.arange(args.eval_way * args.eval_shot)).long().view(1, args.eval_shot, args.eval_way),
                torch.Tensor(np.arange(args.eval_way * args.eval_shot,
                                       args.eval_way * (args.eval_shot + args.eval_query))).long().view(1,
                                                                                                        args.eval_query,
                                                                                                        args.eval_way))

    def split_shot_query(self, data, que, ep_per_batch=1):
        args = self.args  
        img_shape = data.shape[1:]
        data = data.view(ep_per_batch, args.way, args.shot + args.query, *img_shape)
        x_shot, x_query = data.split([args.shot, args.query], dim=2)
        x_shot = x_shot.contiguous()
        x_query = x_query.contiguous().view(ep_per_batch, args.way * args.query, *img_shape)

        que_shape = que.shape[1:]
        que = que.view(ep_per_batch, args.way, args.shot + args.query, *que_shape)
        que_shot, que_query = que.split([args.shot, args.query], dim=2)
        que_shot = que_shot.contiguous()
        que_query = que_query.contiguous().view(ep_per_batch, args.way * args.query, *que_shape)
        return x_shot, x_query, que_shot, que_query

    def forward(self, x, que, support_labels):
   
        x_shot, x_query, que_shot, que_query = self.split_shot_query(x, que, self.args.batch)

        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]
        que_shape = que_shot.shape[-1:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        
        que_shot = que_shot.view(-1, *que_shape)
        que_query = que_query.view(-1, *que_shape)
        
        multi_tot, recon_loss = self.encoder(torch.cat([x_shot, x_query], dim=0), torch.cat([que_shot, que_query], dim=0))

        feat_shape = multi_tot.shape[1:]

        x_shot, x_query = multi_tot[:len(x_shot)], multi_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, *feat_shape)
        x_query = x_query.view(*query_shape, *feat_shape)

        logits = self._forward(x_shot, x_query)
        return logits, recon_loss

    def _forward(self, x_shot, x_query):
        raise NotImplementedError('Suppose to be implemented by subclass')


if __name__ == '__main__':
    pass
