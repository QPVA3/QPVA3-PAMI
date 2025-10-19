import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.EncoderQns import EncoderQns
from .utils.EncoderRNN import EncoderVidHGA
from .utils.graph import GCN
from .utils.cmatt import CMAtten


def padding_mask_k(seq_q, seq_k):
    """ To mask invaild k(all dim are 0), and assign -inf in softmax, seq_k of shape (batch, k_len, k_feat) and seq_q (batch, q_len, q_feat). q and k are padded with 0. pad_mask is (batch, q_len, k_len).
    In batch 0:
    [[x x x 0]     [[0 0 0 1]
     [x x x 0]->    [0 0 0 1]
     [x x x 0]]     [0 0 0 1]] uint8
    """
    fake_q = torch.ones_like(seq_q)
    pad_mask = torch.bmm(fake_q, seq_k.transpose(1, 2))
    pad_mask = pad_mask.eq(0)
    # pad_mask = pad_mask.lt(1e-3)
    return pad_mask


def padding_mask_q(seq_q, seq_k):
    """ To mask invalid q(all dim are 0), seq_k of shape (batch, k_len, k_feat) and seq_q (batch, q_len, q_feat). q and k are padded with 0. pad_mask is (batch, q_len, k_len).
    In batch 0:
    [[x x x x]      [[0 0 0 0]
     [x x x x]  ->   [0 0 0 0]
     [0 0 0 0]]      [1 1 1 1]] uint8
    """
    fake_k = torch.ones_like(seq_k)
    pad_mask = torch.bmm(seq_q, fake_k.transpose(1, 2))
    pad_mask = pad_mask.eq(0)
    # pad_mask = pad_mask.lt(1e-3)
    return pad_mask


def get_u_tile(cls, s, s2):
    """
    attended vectors of s2 for each word in s1,
    signify which words in s2 are most relevant to words in s1
    """
    a_weight = F.softmax(s, dim=2)  # [B, l1, l2]

    a_weight.data.masked_fill_(a_weight.data != a_weight.data, 0)
    # [B, l1, l2] * [B, l2, D] -> [B, l1, D]
    u_tile = torch.bmm(a_weight, s2)
    return u_tile, a_weight


def forward(self, s1, l1, s2, l2):
    s = self.similarity(s1, l1, s2, l2)
    u_tile, a_weight = self.get_u_tile(s, s2)

    return u_tile, a_weight


class AttentionScore(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=-1)

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, scale=None, attn_mask=None, softmax_mask=None):
        """
        Args:
            q: [B, L_q, D_q]
            k: [B, L_k, D_k]
            v: [B, L_v, D_v]
        Return: Same shape to q, but in 'v' space, soft knn
        """

        if attn_mask is None:
            attn_mask = padding_mask_k(q, k)
        if softmax_mask is None:
            softmax_mask = padding_mask_q(q, k)

        # linear projection
        q = self.linear_q(q)
        k = self.linear_k(k)

        scale = q.size(-1) ** -0.5

        attention = torch.bmm(q, k.transpose(-2, -1))

        if scale is not None:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill(attn_mask.bool(), -float("inf"))
        attention = self.softmax(attention)
        attention = attention.masked_fill(softmax_mask, 0.)

        return attention


class VideoAligner(nn.Module):
    ...

class HierarchyVideoAligner(VideoAligner):
    def __init__(self, feat_dim, num_clip, num_frame, num_bbox, hidden_size, input_dropout_p=0.3, tau=1,
                 share_encoder=False):
        super(HierarchyVideoAligner, self).__init__()
        self.dim_feat = feat_dim
        self.num_clip = num_clip
        self.num_bbox = num_bbox
        self.num_frame = num_frame
        self.hidden_size = hidden_size * 2 if not share_encoder else hidden_size
        self.input_dropout_p = input_dropout_p
        self.tau = tau
        self.fg_att = AttentionScore(self.hidden_size, input_dropout_p)
        self.bg_att = AttentionScore(self.hidden_size, input_dropout_p)
        self.share_encoder = share_encoder
        if not share_encoder:
            self.qns_encoder = EncoderQns(768, hidden_size, n_layers=1, rnn_dropout_p=0,
                                          input_dropout_p=input_dropout_p, bidirectional=True, rnn_cell='gru')

            self.app_conv = nn.Sequential(
                nn.Conv1d(feat_dim, self.hidden_size, 3, padding=1),
                nn.ELU(inplace=True)
            )

            self.mot_conv = nn.Sequential(
                nn.Conv1d(feat_dim, self.hidden_size, 3, padding=1),
                nn.ELU(inplace=True)
            )

            self.app_box_conv = nn.Sequential(
                nn.Linear(feat_dim, self.hidden_size),
                nn.ELU(inplace=True)
            )

            self.gcn_hierarchy_1 = GCN(self.hidden_size, self.hidden_size // 2, self.hidden_size,
                                       dropout=input_dropout_p,
                                       skip=True, num_layers=1)
            self.gcn_hierarchy_2 = GCN(self.hidden_size, self.hidden_size // 2, self.hidden_size,
                                       dropout=input_dropout_p,
                                       skip=True, num_layers=1)
            self.gcn_hierarchy_3 = GCN(self.hidden_size, self.hidden_size // 2, self.hidden_size,
                                       dropout=input_dropout_p,
                                       skip=True, num_layers=1)

            self.gcn_atten_pool_hierarchy_1 = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.Tanh(),
                nn.Linear(self.hidden_size // 2, 1),
                nn.Softmax(dim=-2))
            self.gcn_atten_pool_hierarchy_2 = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.Tanh(),
                nn.Linear(self.hidden_size // 2, 1),
                nn.Softmax(dim=-2))
            self.gcn_atten_pool_hierarchy_3 = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.Tanh(),
                nn.Linear(self.hidden_size // 2, 1),
                nn.Softmax(dim=-2))

            self.merge_rf = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.ELU(inplace=True)
            )

            self.merge = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.ELU(inplace=True)
            )

    def forward(self, vid_feats, q_feats, subq_nums, attn_mask=None, backbone=None):
        if self.share_encoder:
            _, vid_feat = backbone(vid_feats, None, q_feats)
            q_l, q_g = backbone.encode_questions(q_feats, None)
            vid_feat = vid_feat.unsqueeze(1)
        else:
            _, vid_feat = self.vid_encode(vid_feats)
            vid_feat = torch.repeat_interleave(vid_feat, subq_nums, dim=0)
            qns, ques_lengths = q_feats
            q_l, q_g = self.qns_encoder(qns, ques_lengths)

        fg_score = self.fg_att(q_g.unsqueeze(1), vid_feat, attn_mask=attn_mask)  # [bs, 1, 16]
        bg_score = self.bg_att(q_g.unsqueeze(1), vid_feat, attn_mask=attn_mask)
        score = torch.cat((fg_score, bg_score), 1)  # [bs, 2, 16]
        score = F.gumbel_softmax(score, tau=self.tau, hard=True, dim=1)  # [bs, 2, 16]

        fg_mask = score[:, 0, :]  # [bs, 16]
        bg_mask = score[:, 1, :]  # [bs, 16]
        return fg_mask, bg_mask

    def vid_encode(self, vid_feats):
        app_feat, mot_feat, app_box_feat = vid_feats

        batch_size, num_clip, frame_pclip, box_pframe, dim = app_box_feat.shape

        mot_feat = self.mot_conv(mot_feat.transpose(1, 2)).transpose(1, 2)

        app_feat = app_feat.reshape(batch_size, -1, dim)
        app_feat = self.app_conv(app_feat.transpose(1, 2)).transpose(1, 2)
        app_feat = app_feat.reshape(batch_size, num_clip, frame_pclip, -1)

        app_box_feat = self.app_box_conv(app_box_feat)

        vid_feat_hidden = self.hierarchy([app_feat, mot_feat, app_box_feat])

        return (app_box_feat, app_feat, mot_feat), vid_feat_hidden

    def hierarchy(self, vid_feats):
        app_feats, mot_feats, app_box_feat = vid_feats
        batch_size, num_clip, frame_pclip, region_pframe, feat_dim = app_box_feat.size()

        #############app_box########################
        xlen = num_clip * frame_pclip * region_pframe
        app_box_feat = app_box_feat.reshape(batch_size, xlen, feat_dim)

        app_box_feat = app_box_feat.reshape(batch_size, num_clip, frame_pclip, region_pframe, feat_dim)
        app_box_feat = app_box_feat.reshape(-1, region_pframe, feat_dim)
        num_rpframe = torch.tensor([region_pframe] * app_box_feat.shape[0], dtype=torch.long)

        gcn_output_app_box, _ = self.gcn_hierarchy_1(app_box_feat, num_rpframe)
        att_app_box = self.gcn_atten_pool_hierarchy_1(gcn_output_app_box)
        gcn_att_pool_app_box = torch.sum(gcn_output_app_box * att_app_box, dim=1)
        gcn_att_pool_app_box = gcn_att_pool_app_box.reshape(batch_size, num_clip, frame_pclip, -1)

        #############app+app_box########################
        gcn_att_pool_app_box = gcn_att_pool_app_box.reshape(batch_size, num_clip * frame_pclip, -1)
        app_feats = app_feats.reshape(batch_size, num_clip * frame_pclip, -1)
        tmp = torch.cat((app_feats, gcn_att_pool_app_box), -1)
        gcn_app_feat = self.merge_rf(tmp)

        gcn_app_feat = gcn_app_feat.reshape(-1, frame_pclip, feat_dim)
        num_fpclip = torch.tensor([frame_pclip] * gcn_app_feat.shape[0], dtype=torch.long)
        gcn_output_app, _ = self.gcn_hierarchy_2(gcn_app_feat, num_fpclip)
        att_app = self.gcn_atten_pool_hierarchy_2(gcn_output_app)
        gcn_att_pool_app = torch.sum(gcn_output_app * att_app, dim=1)
        gcn_att_pool_app = gcn_att_pool_app.reshape(batch_size, num_clip, -1)

        #############mot+app########################
        tmp = torch.cat((mot_feats, gcn_att_pool_app), -1)
        gcn_mot_app_feat = self.merge_rf(tmp)

        num_cpvideo = torch.tensor([num_clip] * gcn_mot_app_feat.shape[0], dtype=torch.long)
        gcn_output_mot_app, _ = self.gcn_hierarchy_3(gcn_mot_app_feat, num_cpvideo)  # （batch_size, seq_len, dim_hidden）
        att = self.gcn_atten_pool_hierarchy_3(gcn_output_mot_app)
        gcn_att_pool_mot_app = torch.sum(gcn_output_mot_app * att, dim=1, keepdim=True)
        gcn_att_pool_mot_app = gcn_att_pool_mot_app + gcn_output_mot_app  # gcn_att_pool_mot_app.reshape(batch_size, -1)

        return gcn_att_pool_mot_app

