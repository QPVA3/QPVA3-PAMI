import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import rnn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import os

from .graph import GCN

def init_modules(modules, w_init='kaiming_uniform'):
    if w_init == "normal":
        _init = init.normal_
    elif w_init == "xavier_normal":
        _init = init.xavier_normal_
    elif w_init == "xavier_uniform":
        _init = init.xavier_uniform_
    elif w_init == "kaiming_normal":
        _init = init.kaiming_normal_
    elif w_init == "kaiming_uniform":
        _init = init.kaiming_uniform_
    elif w_init == "orthogonal":
        _init = init.orthogonal_
    else:
        raise NotImplementedError
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            _init(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        if isinstance(m, (nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.zeros_(param)
                elif 'weight' in name:
                    _init(param)

class EncoderQns(nn.Module):
    def __init__(self, dim_embed, dim_hidden, vocab_size, glove_embed, use_bert=True, input_dropout_p=0.2, rnn_dropout_p=0.1, n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        """
        super(EncoderQns, self).__init__()
        self.dim_hidden = dim_hidden
        self.vocab_size = vocab_size
        self.glove_embed = glove_embed
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        self.input_dropout = nn.Dropout(input_dropout_p)
        self.rnn_dropout = nn.Dropout(rnn_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        input_dim = dim_embed
        self.use_bert = use_bert
        if self.use_bert:
            self.embedding = nn.Linear(input_dim, dim_embed, bias=True)
        else:
            self.embedding = nn.Embedding(vocab_size, dim_embed)

        self.obj_embedding = nn.Linear(2048, dim_embed, bias=False)

        self.rnn = self.rnn_cell(dim_embed, dim_hidden//2, n_layers, batch_first=True,
                                bidirectional=bidirectional)

        # init_modules(self.modules(), w_init="xavier_uniform")
        # nn.init.uniform_(self.embedding.weight, -1.0, 1.0)

        if not self.use_bert and os.path.exists(self.glove_embed):
            word_mat = torch.FloatTensor(np.load(self.glove_embed))
            self.embedding = nn.Embedding.from_pretrained(word_mat, freeze=False)

    def forward(self, qns, qns_lengths, hidden=None, obj=None):
        """
         encode question
        :param qns:
        :param qns_lengths:
        :return:
        """
        qns_embed = self.embedding(qns)
        if obj is not None:
            obj_embed = self.obj_embedding(obj)
            qns_embed = qns_embed + obj_embed
        qns_embed = self.input_dropout(qns_embed)
        packed = pack_padded_sequence(qns_embed, qns_lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed, hidden)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        hidden = hidden.permute(1, 0, 2).contiguous().view(output.shape[0], -1)
        return output, hidden


class EncoderVid(nn.Module):
    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        """
        super(EncoderVid, self).__init__()
        self.dim_vid = dim_vid
        self.dim_app = 2048
        self.dim_motion = 4096
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn = self.rnn_cell(dim_vid, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)


    def forward(self, vid_feats):

        self.rnn.flatten_parameters()
        foutput, fhidden = self.rnn(vid_feats)

        return foutput, fhidden


class EncoderVidSTVQA(nn.Module):
    def __init__(self, input_dim, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        """
        super(EncoderVidSTVQA, self).__init__()
        self.input_dim = input_dim
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell


        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn1 = self.rnn_cell(input_dim, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self.rnn2 = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                 bidirectional=bidirectional, dropout=self.rnn_dropout_p)


    def forward(self, vid_feats):
        """
        Dual-layer LSTM
        """

        self.rnn1.flatten_parameters()

        foutput_1, fhidden_1 = self.rnn1(vid_feats)
        self.rnn2.flatten_parameters()
        foutput_2, fhidden_2 = self.rnn2(foutput_1)

        foutput = torch.cat((foutput_1, foutput_2), dim=2)
        fhidden = (torch.cat((fhidden_1[0], fhidden_2[0]), dim=0),
                   torch.cat((fhidden_1[1], fhidden_2[1]), dim=0))

        return foutput, fhidden


class EncoderVidCoMem(nn.Module):
    def __init__(self, dim_app, dim_motion, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=2, bidirectional=False, rnn_cell='gru'):
        """
        """
        super(EncoderVidCoMem, self).__init__()
        self.dim_app = dim_app
        self.dim_motion = dim_motion
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn_app_l1 = self.rnn_cell(self.dim_app, dim_hidden, n_layers, batch_first=True,
                                        bidirectional=bidirectional, dropout=self.rnn_dropout_p)
        self.rnn_app_l2 = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                        bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self.rnn_motion_l1 = self.rnn_cell(self.dim_motion, dim_hidden, n_layers, batch_first=True,
                                            bidirectional=bidirectional, dropout=self.rnn_dropout_p)
        self.rnn_motion_l2 = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                           bidirectional=bidirectional, dropout=self.rnn_dropout_p)


    def forward(self, vid_feats, temporal_mask=None):
        """
        two separate LSTM to encode app and motion feature
        :param vid_feats:
        :return:
        """
        if temporal_mask is not None:
            # stack selected features to left
            temporal_mask_ = temporal_mask.bool().clone()
            temporal_mask_[temporal_mask_.sum(-1) == 0] = 1 # if no fg frame, manualy set allframe to be fg
            fg_len = temporal_mask_.sum(-1)
            flatten_fg_features = vid_feats[temporal_mask_]
            assert flatten_fg_features.size(0) == fg_len.sum()
            fg_feature_list = torch.split(flatten_fg_features, fg_len.tolist(), dim=0)
            temp = rnn.pad_sequence(fg_feature_list, batch_first=True)
            vid_feats = pack_padded_sequence(temp, temporal_mask_.cpu().sum(-1), batch_first=True, enforce_sorted=False)
        
        vid_app = vid_feats[:, :, 0:self.dim_app]
        vid_motion = vid_feats[:, :, self.dim_app:]

        app_output_l1, app_hidden_l1 = self.rnn_app_l1(vid_app)
        app_output_l2, app_hidden_l2 = self.rnn_app_l2(app_output_l1)


        motion_output_l1, motion_hidden_l1 = self.rnn_motion_l1(vid_motion)
        motion_output_l2, motion_hidden_l2 = self.rnn_motion_l2(motion_output_l1)

        return app_output_l1, app_output_l2, motion_output_l1, motion_output_l2


class EncoderVidHGA(nn.Module):
    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=True, rnn_cell='gru'):
        """
        """
        super(EncoderVidHGA, self).__init__()
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        self.v_input_ln = nn.LayerNorm((dim_hidden*2 if bidirectional else dim_hidden), elementwise_affine=False)
        self.vid2hid = nn.Sequential(nn.Linear(self.dim_vid, dim_hidden),
                                     nn.ReLU(),
                                     nn.Dropout(input_dropout_p))


        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self._init_weight()


    def _init_weight(self):
        nn.init.xavier_normal_(self.vid2hid[0].weight)


    def forward(self, vid_feats, temporal_mask=None):
        """
        """
        batch_size, seq_len, dim_vid = vid_feats.size()
        vid_feats_trans = self.vid2hid(vid_feats.reshape(-1, self.dim_vid))
        vid_feats = vid_feats_trans.reshape(batch_size, seq_len, -1)

        if temporal_mask is not None:
            # stack selected features to left
            temporal_mask_ = temporal_mask.bool().clone()
            temporal_mask_[temporal_mask_.sum(-1) == 0] = 1 # if no fg frame, manualy set allframe to be fg
            fg_len = temporal_mask_.sum(-1)
            flatten_fg_features = vid_feats[temporal_mask_]
            assert flatten_fg_features.size(0) == fg_len.sum()
            fg_feature_list = torch.split(flatten_fg_features, fg_len.tolist(), dim=0)
            temp = rnn.pad_sequence(fg_feature_list, batch_first=True)
            vid_feats = pack_padded_sequence(temp, temporal_mask_.cpu().sum(-1), batch_first=True, enforce_sorted=False)

        # self.rnn.flatten_parameters()
        foutput, fhidden = self.rnn(vid_feats)

        if self.bidirectional:
            fhidden = torch.cat([fhidden[0], fhidden[1]], dim=-1)
        else:
            fhidden = torch.squeeze(fhidden, 0)

        if temporal_mask is not None:
            foutput, _ = pad_packed_sequence(foutput, batch_first=True)
        foutput = self.v_input_ln(foutput) # bs,16,hidden_dim

        if temporal_mask is not None:
            return foutput, fhidden, temporal_mask_.to(foutput.dtype)
        return foutput, fhidden

class EncoderVidB2A(nn.Module):
    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        """
        super(EncoderVidB2A, self).__init__()
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell


        self.vid2hid = nn.Sequential(nn.Linear(self.dim_vid, dim_hidden),
                                     nn.ReLU(),
                                     nn.Dropout(input_dropout_p))


        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self._init_weight()


    def _init_weight(self):
        nn.init.xavier_normal_(self.vid2hid[0].weight)


    def forward(self, app_feat, mot_feat):
        """
        """
        batch_size, seq_len, seq_len2, dim_vid = app_feat.size()

        app_feat_trans = self.vid2hid(app_feat.reshape(-1, self.dim_vid))
        app_feat = app_feat_trans.reshape(batch_size, seq_len*seq_len2, -1)

        mot_feat_trans = self.vid2hid(mot_feat.reshape(-1, self.dim_vid))
        mot_feat = mot_feat_trans.reshape(batch_size, seq_len, -1)

        self.rnn.flatten_parameters()
        app_output, _ = self.rnn(app_feat)
        mot_output, _ = self.rnn(mot_feat)

        return app_output, mot_output

class EncoderVidCAU(nn.Module):
    def __init__(self, feat_dim, num_clip, num_frame, num_bbox, feat_hidden, input_dropout_p=0.3):
        
        super(EncoderVidCAU, self).__init__()
        self.dim_feat = feat_dim
        self.num_clip = num_clip
        self.num_bbox = num_bbox
        self.num_frame = num_frame
        self.dim_hidden = feat_hidden
        self.input_dropout_p = input_dropout_p

        self.app_conv = nn.Sequential(
            nn.Conv1d(feat_dim, self.dim_hidden, 3, padding=1),
            nn.ELU(inplace=True)
        )

        self.mot_conv = nn.Sequential(
            nn.Conv1d(feat_dim, self.dim_hidden, 3, padding=1),
            nn.ELU(inplace=True)
        )

        self.app_box_conv = nn.Sequential(
            nn.Conv2d(feat_dim, self.dim_hidden, (3, 1), padding=(1, 0)),
            nn.ELU(inplace=True)
        )

        self.mot_box_conv = nn.Sequential(
            nn.Conv2d(feat_dim, self.dim_hidden, (3, 1), padding=(1, 0)),
            nn.ELU(inplace=True)
        )

        self.gcn_hierarchy = GCN(self.dim_hidden, self.dim_hidden//2, self.dim_hidden, dropout=input_dropout_p, skip=True, num_layers=1)

        self.gcn_atten_pool_hierarchy = nn.Sequential(
            nn.Linear(self.dim_hidden, self.dim_hidden // 2),
            nn.Tanh(),
            nn.Linear(self.dim_hidden // 2, 1),
            nn.Softmax(dim=-1))
        
        self.merge_rf = nn.Sequential(
            nn.Linear(self.dim_hidden * 2, self.dim_hidden),
            nn.ELU(inplace=True)
        )

        self.merge = nn.Sequential(
            nn.Linear(self.dim_hidden * 2, self.dim_hidden),
            nn.ELU(inplace=True)
        )

    def forward(self, vid_feats):
        app_feat, mot_feat, app_box_feat, mot_box_feat = vid_feats
        
        batch_size, num_clip, frame_pclip, box_pframe, dim = app_box_feat.shape

        app_feat = app_feat.reshape(batch_size, -1, dim)
        app_feat = self.app_conv(app_feat.transpose(1, 2)).transpose(1, 2)
        app_feat = app_feat.reshape(batch_size, num_clip, frame_pclip, -1)
        
        app_box_feat = app_box_feat.reshape(batch_size, -1, box_pframe, dim)
        app_box_feat = self.app_box_conv(app_box_feat.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        app_box_feat = app_box_feat.reshape(batch_size, num_clip, frame_pclip, box_pframe, -1)
        
        mot_feat = self.mot_conv(mot_feat.transpose(1, 2)).transpose(1, 2)

        mot_box_feat = self.app_box_conv(mot_box_feat.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        vid_feat_seq = self.hierarchy([app_feat, mot_feat, app_box_feat, mot_box_feat])
                
        return (app_feat, mot_feat, app_box_feat, mot_box_feat), vid_feat_seq

    def hierarchy(self, vid_feats):
        app_feats, mot_feats, app_box_feat, mot_box_feat = vid_feats
        batch_size, num_clip, frame_pclip, region_pframe, feat_dim = app_box_feat.size()

        #############app_box########################
        xlen = num_clip*frame_pclip*region_pframe
        app_box_feat = app_box_feat.reshape(batch_size, xlen, feat_dim)

        app_box_feat = app_box_feat.reshape(batch_size, num_clip, frame_pclip, region_pframe, feat_dim)
        app_box_feat = app_box_feat.reshape(-1, region_pframe, feat_dim)
        num_rpframe = torch.tensor([region_pframe] * app_box_feat.shape[0], dtype=torch.long)
        
        gcn_output_app_box, _ = self.gcn_hierarchy(app_box_feat, num_rpframe)
        att_app_box = self.gcn_atten_pool_hierarchy(gcn_output_app_box)
        gcn_att_pool_app_box = torch.sum(gcn_output_app_box*att_app_box, dim=1)
        gcn_att_pool_app_box = gcn_att_pool_app_box.reshape(batch_size, num_clip, frame_pclip, -1)

        #############mot_box########################
        xlen = num_clip*region_pframe
        mot_box_feat = mot_box_feat.reshape(batch_size, xlen, feat_dim)

        mot_box_feat = mot_box_feat.reshape(-1, region_pframe, feat_dim)
        num_rpframe = torch.tensor([region_pframe] * mot_box_feat.shape[0], dtype=torch.long)
        
        gcn_output_mot_box, _ = self.gcn_hierarchy(mot_box_feat, num_rpframe)
        att_mot_box = self.gcn_atten_pool_hierarchy(gcn_output_mot_box)
        gcn_att_pool_mot_box = torch.sum(gcn_output_mot_box*att_mot_box, dim=1)
        gcn_att_pool_mot_box = gcn_att_pool_mot_box.reshape(batch_size, num_clip, -1)

        #############app+app_box########################
        gcn_att_pool_app_box = gcn_att_pool_app_box.reshape(batch_size, num_clip*frame_pclip, -1)
        app_feats = app_feats.reshape(batch_size, num_clip*frame_pclip, -1)
        tmp = torch.cat((app_feats, gcn_att_pool_app_box), -1)
        gcn_app_feat = self.merge_rf(tmp)
        
        gcn_app_feat = gcn_app_feat.reshape(-1, frame_pclip, feat_dim)
        num_fpclip = torch.tensor([frame_pclip] * gcn_app_feat.shape[0], dtype=torch.long)
        gcn_output_app, _ = self.gcn_hierarchy(gcn_app_feat, num_fpclip)
        att_app = self.gcn_atten_pool_hierarchy(gcn_output_app)
        gcn_att_pool_app = torch.sum(gcn_output_app * att_app, dim=1)
        gcn_att_pool_app = gcn_att_pool_app.reshape(batch_size, num_clip, -1)

        #############mot+mot_box########################
        tmp = torch.cat((mot_feats, gcn_att_pool_mot_box), -1)
        gcn_mot_feat = self.merge_rf(tmp)

        num_cpvideo = torch.tensor([num_clip] * gcn_mot_feat.shape[0], dtype=torch.long)
        gcn_output_mot, _ = self.gcn_hierarchy(gcn_mot_feat, num_cpvideo)

        #############mot+app########################
        tmp = torch.cat((mot_feats, gcn_att_pool_app), -1)
        gcn_mot_app_feat = self.merge_rf(tmp)

        num_cpvideo = torch.tensor([num_clip] * gcn_mot_feat.shape[0], dtype=torch.long)
        gcn_output_mot_app, _ = self.gcn_hierarchy(gcn_mot_app_feat, num_cpvideo)

        visual_out = self.merge(torch.cat([gcn_output_mot, gcn_output_mot_app], dim=-1))

        return visual_out

class EncoderVidCAU2(nn.Module):
    def __init__(self, feat_dim, num_clip, num_frame, num_bbox, feat_hidden, input_dropout_p=0.3):
        
        super(EncoderVidCAU2, self).__init__()
        self.dim_feat = feat_dim
        self.num_clip = num_clip
        self.num_bbox = num_bbox
        self.num_frame = num_frame
        self.dim_hidden = feat_hidden
        self.input_dropout_p = input_dropout_p

        self.app_conv = nn.Sequential(
            nn.Conv1d(feat_dim, self.dim_hidden, 3, padding=1),
            nn.ELU(inplace=True)
        )

        self.mot_conv = nn.Sequential(
            nn.Conv1d(feat_dim, self.dim_hidden, 3, padding=1),
            nn.ELU(inplace=True)
        )

        self.app_box_conv = nn.Sequential(
            nn.Linear(feat_dim, self.dim_hidden),
            nn.ELU(inplace=True)
        )

        # self.gcn_hierarchy_1 = GCN(self.dim_hidden, self.dim_hidden//2, self.dim_hidden, dropout=input_dropout_p, skip=True, num_layers=2)
        # self.gcn_hierarchy_2 = GCN(self.dim_hidden, self.dim_hidden//2, self.dim_hidden, dropout=input_dropout_p, skip=True, num_layers=2)
        # self.gcn_hierarchy_3 = GCN(self.dim_hidden, self.dim_hidden//2, self.dim_hidden, dropout=input_dropout_p, skip=True, num_layers=2)
        #
        # self.gcn_atten_pool_hierarchy_1 = nn.Sequential(
        #     nn.Linear(self.dim_hidden, self.dim_hidden // 2),
        #     nn.Tanh(),
        #     nn.Linear(self.dim_hidden // 2, 1),
        #     nn.Softmax(dim=-2))
        # self.gcn_atten_pool_hierarchy_2 = nn.Sequential(
        #     nn.Linear(self.dim_hidden, self.dim_hidden // 2),
        #     nn.Tanh(),
        #     nn.Linear(self.dim_hidden // 2, 1),
        #     nn.Softmax(dim=-2))
        # self.gcn_atten_pool_hierarchy_3 = nn.Sequential(
        #     nn.Linear(self.dim_hidden, self.dim_hidden // 2),
        #     nn.Tanh(),
        #     nn.Linear(self.dim_hidden // 2, 1),
        #     nn.Softmax(dim=-2))
        #
        # self.merge_rf = nn.Sequential(
        #     nn.Linear(self.dim_hidden * 2, self.dim_hidden),
        #     nn.ELU(inplace=True)
        # )
        #
        # self.merge = nn.Sequential(
        #     nn.Linear(self.dim_hidden * 2, self.dim_hidden),
        #     nn.ELU(inplace=True)
        # )

    def forward(self, vid_feats):
        app_box_feat, app_feat, mot_feat  = vid_feats
        
        batch_size, num_clip, frame_pclip, box_pframe, dim = app_box_feat.shape

        mot_feat = self.mot_conv(mot_feat.transpose(1, 2)).transpose(1, 2)

        app_feat = app_feat.reshape(batch_size, -1, dim)
        app_feat = self.app_conv(app_feat.transpose(1, 2)).transpose(1, 2)
        app_feat = app_feat.reshape(batch_size, num_clip, frame_pclip, -1)
        
        app_box_feat = self.app_box_conv(app_box_feat)

        # vid_feat_hidden = self.hierarchy([app_feat, mot_feat, app_box_feat])

        return (app_box_feat, app_feat, mot_feat), None#vid_feat_hidden

    def hierarchy(self, vid_feats):
        app_feats, mot_feats, app_box_feat = vid_feats
        batch_size, num_clip, frame_pclip, region_pframe, feat_dim = app_box_feat.size()

        #############app_box########################
        xlen = num_clip*frame_pclip*region_pframe
        app_box_feat = app_box_feat.reshape(batch_size, xlen, feat_dim)

        app_box_feat = app_box_feat.reshape(batch_size, num_clip, frame_pclip, region_pframe, feat_dim)
        app_box_feat = app_box_feat.reshape(-1, region_pframe, feat_dim)
        num_rpframe = torch.tensor([region_pframe] * app_box_feat.shape[0], dtype=torch.long)
        
        gcn_output_app_box, _ = self.gcn_hierarchy_1(app_box_feat, num_rpframe)
        att_app_box = self.gcn_atten_pool_hierarchy_1(gcn_output_app_box)
        gcn_att_pool_app_box = torch.sum(gcn_output_app_box*att_app_box, dim=1)
        gcn_att_pool_app_box = gcn_att_pool_app_box.reshape(batch_size, num_clip, frame_pclip, -1)

        #############app+app_box########################
        gcn_att_pool_app_box = gcn_att_pool_app_box.reshape(batch_size, num_clip*frame_pclip, -1)
        app_feats = app_feats.reshape(batch_size, num_clip*frame_pclip, -1)
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
        gcn_output_mot_app, _ = self.gcn_hierarchy_3(gcn_mot_app_feat, num_cpvideo) # （batch_size, seq_len, dim_hidden）
        att = self.gcn_atten_pool_hierarchy_3(gcn_output_mot_app)
        gcn_att_pool_mot_app = torch.sum(gcn_output_mot_app * att, dim=1)
        gcn_att_pool_mot_app = gcn_att_pool_mot_app.reshape(batch_size, -1)

        return gcn_att_pool_mot_app