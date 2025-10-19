from torch import nn
import torch


class EncoderQns(nn.Module):
    def __init__(self, dim_embed, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):

        super(EncoderQns, self).__init__()
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell
        self.q_input_ln = nn.LayerNorm((dim_hidden * 2 if bidirectional else dim_hidden), elementwise_affine=False)
        self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        # self.embedding = nn.Linear(768, dim_embed)
        self.embedding = nn.Sequential(nn.Linear(dim_embed, dim_embed),
                                       nn.ReLU(),
                                       nn.Dropout(input_dropout_p))

        self.rnn = self.rnn_cell(dim_embed, dim_hidden, n_layers, batch_first=True,
                                 bidirectional=bidirectional, dropout=self.rnn_dropout_p)
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_normal_(self.embedding[0].weight)

    def forward(self, qns, qns_lengths):
        """
         encode question
        :param qns:
        :param qns_lengths:
        :return:
        """
        qns_embed = self.embedding(qns)
        qns_embed = self.input_dropout(qns_embed)
        packed = nn.utils.rnn.pack_padded_sequence(qns_embed, qns_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        if self.bidirectional:
            hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        else:
            hidden = hidden[0]
        output = self.q_input_ln(output)  # bs,q_len,hidden_dim

        return output, hidden
