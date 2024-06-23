import torch
import torch.nn as nn
import numpy as np
from model.networks.lstm import LSTM
from model.networks.transformer_entropy import Transformer
# from model.networks.transformer_entropy_deatt import Transformer
# from model.networks.transformer_entropy_aoa import Transformer


class VQABackbone(nn.Module):
    def __init__(self, args, hidden_dim=768, pre_train=False):
        super().__init__()
        self.args = args
        self.pre_train = pre_train

        # resnet12 => 640
        if args.backbone_class == 'Res12':
            from model.networks.res12 import ResNet
            # self.encoder = ResNet()
            self.encoder = ResNet(avg_pool=False)
            args.hidden_dim = 640
        elif args.backbone_class == 'SwinT':
            from model.networks.swin_transformer import SwinTransformer
            self.encoder = SwinTransformer(window_size=7, embed_dim=96, depths=[2, 2, 6, 2], 
                            num_heads=[3, 6, 12, 24],mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1)
            args.hidden_dim = 768
        elif args.backbone_class == 'VitS':
            from model.networks.vision_transformer import VisionTransformer
            self.encoder = VisionTransformer(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True)
            args.hidden_dim = 384
        else:
            raise ValueError('')
        
        # self.que_encoder = LSTM(args.pretrained_emb, args.token_size)
        self.que_encoder = LSTM(args.pretrained_emb, args.token_size, hidden_dim=args.hidden_dim, avg_pool=False)
        self.transformer = Transformer(hidden_dim=args.hidden_dim)

        if pre_train:
            self.multi_linear = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
            self.final = nn.Linear(args.hidden_dim, args.num_class)
        else:
            self.multi_linear_proto = nn.Linear(args.hidden_dim * 2, args.hidden_dim)

    def forward(self, img, que):
     
        
        if self.args.backbone_class in ['VitS', 'SwinT']:
            img_tot = self.encoder.forward(img, return_all_tokens=True)[:, 1:]
        else:
            img_tot = self.encoder(img)
        img_mask = self.make_mask(img_tot)

        que_mask = self.make_mask(que.unsqueeze(2))
        que_tot = self.que_encoder(que)

        multi_tot, recon_loss = self.transformer(img_tot, que_tot, img_mask, que_mask)
        
        if self.pre_train:
            multi_tot = self.multi_linear(multi_tot)
            multi_tot = self.final(multi_tot)
        else:
            multi_tot = self.multi_linear_proto(multi_tot)

        return multi_tot, recon_loss


    def make_mask(self, feature):
        return (torch.sum(
                torch.abs(feature),
                dim=-1
            ) == 0).unsqueeze(1).unsqueeze(2)
