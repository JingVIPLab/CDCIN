

import torch.nn as nn
import torch.nn.functional as F
import torch, math


class MHAtt(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
        super(MHAtt, self).__init__()
        self.head = head
        self.hidden_dim = hidden_dim
        self.head_size = int(hidden_dim / 8)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_q = nn.Linear(hidden_dim,hidden_dim)
        self.linear_merge = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_r)

    def forward(self, v, k, q, mask=None):
        b, n, s = q.shape

        v = self.linear_v(v).view(b, -1, self.head, self.head_size).transpose(1, 2)
        k = self.linear_k(k).view(b, -1, self.head, self.head_size).transpose(1, 2)
        q = self.linear_q(q).view(b, -1, self.head, self.head_size).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(b, -1, self.hidden_dim)
        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -65504.0)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class PositionWiseFFN(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, outdim=640):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_r)
        self.dense2 = nn.Linear(hidden_dim * 2, outdim)

    def forward(self, X):
        return self.dense2(self.dropout(self.relu(self.dense1(X))))

class AttFlat(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, out_dim=640, glimpses=1):
        super(AttFlat, self).__init__()
        self.glimpses = glimpses

        self.mlp = PositionWiseFFN(hidden_dim, dropout_r, self.glimpses)

        self.linear_merge = nn.Linear(
            hidden_dim * glimpses,
            out_dim
        )

    def forward(self, x, x_mask=None):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -65504.0
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted, att.squeeze()

class Encoder(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
        super(Encoder, self).__init__()

        self.mhatt = MHAtt(hidden_dim, dropout_r, head)
        self.ffn = PositionWiseFFN(hidden_dim, dropout_r, hidden_dim)

        self.dropout1 = nn.Dropout(dropout_r)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.dropout2 = nn.Dropout(dropout_r)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(self.mhatt(x, x, x, x_mask)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))

        return x


class Decoder(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8):
        super(Decoder, self).__init__()

        self.mhatt1 = MHAtt(hidden_dim, dropout_r, head)
        self.mhatt2 = MHAtt(hidden_dim, dropout_r, head)
        self.ffn = PositionWiseFFN(hidden_dim, dropout_r, hidden_dim)

        self.dropout1 = nn.Dropout(dropout_r)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.dropout2 = nn.Dropout(dropout_r)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout3 = nn.Dropout(dropout_r)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(self.mhatt1(x, x, x, x_mask)))

        x = self.norm2(x + self.dropout2(self.mhatt2(y, y, x, y_mask)))
        x = self.norm3(x + self.dropout3(self.ffn(x)))

        return x


class BiAttention(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1):
        super(BiAttention, self).__init__()

        self.hidden_dim = hidden_dim

        self.l_flatten = AttFlat(hidden_dim, dropout_r, hidden_dim)
        self.i_flatten = AttFlat(hidden_dim, dropout_r, hidden_dim)


        self.final = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, i_batch, q_batch, i_mask, q_mask):

        i_feat  = self.qkv_attention(i_batch, q_batch, q_batch, q_mask)
        i_feat, i_weight = self.l_flatten(i_feat, i_mask)

        l_feat = self.qkv_attention(q_batch, i_batch, i_batch, i_mask)
        l_feat, _ = self.i_flatten(l_feat, q_mask)

        return self.final(torch.cat((l_feat, i_feat), dim=-1)), i_weight

    def qkv_attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            scores.data.masked_fill_(mask.squeeze(1), -65504.0)
        
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value)


class AGAttention(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1):
        super(AGAttention, self).__init__()
        self.lin_v = PositionWiseFFN(hidden_dim, dropout_r, hidden_dim)
        self.lin_q = PositionWiseFFN(hidden_dim, dropout_r, hidden_dim)
        self.lin = PositionWiseFFN(hidden_dim, dropout_r, 1)

    def forward(self, v, q, v_mask):
        """
        v = batch, num_obj, dim
        q = batch, dim
        """
        v = self.lin_v(v)
        q = self.lin_q(q)
        batch, num_obj, _ = v.shape
        _, q_dim = q.shape
        q = q.unsqueeze(1).expand(batch, num_obj, q_dim)

        x = v * q
        x = self.lin(x)  # batch, num_obj, glimps

        x = x.squeeze(-1).masked_fill(v_mask.squeeze(2).squeeze(1), -65504.0)

        x = F.softmax(x, dim=1)

        return x


class Transformer(nn.Module):
    def __init__(self, hidden_dim=640, dropout_r=0.1, head=8, avg_pool=True):
        super(Transformer, self).__init__()
        self.avg_pool = avg_pool
        self.entroy_thred = 0.0

        self.enc_list = nn.ModuleList([Encoder(hidden_dim, dropout_r, head) for _ in range(1)])
        self.dec_list = nn.ModuleList([Decoder(hidden_dim, dropout_r, head) for _ in range(1)])

        
        self.bi_attention = BiAttention(hidden_dim, dropout_r)

        self.attflat_que = AttFlat(hidden_dim, dropout_r, hidden_dim * 2)
        self.linear_que = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ag_attention = AGAttention(hidden_dim, dropout_r)

        self.final = nn.Linear(hidden_dim, hidden_dim * 2)
        self.proj_norm = nn.LayerNorm(hidden_dim * 2)

        self.temp = nn.Parameter(torch.tensor(100., requires_grad=True))
        

    def forward(self, img, que, img_mask, que_mask):
        for enc in self.enc_list:
            que = enc(que, que_mask)
        
        b, n, c = img.shape
        img_ori = img
        self.total_num = n
        for dec in self.dec_list:
            img = dec(img, que, img_mask, que_mask)

      
        proj_feat, attn_weight = self.bi_attention(img, que, img_mask, que_mask)

        que, que_weight = self.attflat_que(
            que,
            que_mask
        )

        recon_weight = self.ag_attention(img_ori, proj_feat + self.linear_que(que), img_mask)

        
        entropy_rate = self.com_recon_ent_rate(img_mask, attn_weight, self.entroy_thred)
        recon_loss = self.recon_loss_enhance(attn_weight=attn_weight, recon_weight=recon_weight, entropy_rate=entropy_rate)
        

        proj_feat = self.final(proj_feat)
        proj_feat = self.proj_norm(proj_feat)
        return proj_feat, recon_loss
    
    def com_recon_ent_rate(self, img_mask, learned_weight, recon_thod):
        
        # Make mask
        img_mask_tmp = img_mask.squeeze(1).squeeze(1)
        object_num = self.total_num - torch.sum(img_mask_tmp, dim=-1)
        avg_weight = torch.div(torch.ones_like(learned_weight).float().cuda(), object_num.unsqueeze(1).float())
        entropy_avg = self.get_entropy(avg_weight)

        entropy_attn = self.get_entropy(torch.where(learned_weight==0.0, torch.zeros_like(learned_weight).float().cuda()+1e-9, learned_weight))

        entropy_rate = torch.where(
            torch.div(entropy_attn-entropy_avg, entropy_avg)>recon_thod, 
            torch.ones_like(entropy_avg).float().cuda(), 
            torch.zeros_like(entropy_avg).float().cuda()
            )
        return entropy_rate
    
    def get_entropy(self, data_df):
        return torch.sum(
            (-data_df)*torch.log(data_df), 
            dim=-1
            )

    def recon_loss_enhance(self, attn_weight, recon_weight, entropy_rate):
        error = (attn_weight - recon_weight).view(attn_weight.size(0), -1)
        error = error**2
        error = torch.sum(error, dim=1)  # * 0.0005

        error = torch.sum(torch.mul(error, entropy_rate), dim=-1)

        return error * self.temp

    def recon_loss(self, attn_weight, recon_weight):
        error = (attn_weight - recon_weight).view(attn_weight.size(0), -1)
        error = error**2
        error = torch.sum(error, dim=1)  # * 0.0005

        # Average over batch
        error = error.mean()# * self.__C.recon_rate
        return error


