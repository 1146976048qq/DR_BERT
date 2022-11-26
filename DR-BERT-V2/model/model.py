import torch.nn as nn
import torch.nn.functional as F
import torch
from base.base_model import BaseModel
from utils.model_utils import prepare_pack_padded_sequence, prepare_pack_padded_sequence_asp, function as F
from transformers import BertModel


class MLP(BaseModel):
    """
    MLP
    """

    def __init__(self, hidden_dim, output_dim, dropout, word_embedding, freeze,
                 needed_by_mlp_num_hidden_layers, needed_by_mlp_max_seq_len):
        """
        1、data_process.py，文本提前做了padding
        2、model.py，可选隐藏层数
        """

        super().__init__()

        self.embedding_dim = word_embedding.vectors.shape[1]
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embedding.vectors), freeze=freeze)

        # 隐藏层数
        self.num_hidden_layers = needed_by_mlp_num_hidden_layers
        # 最大文本次长度
        self.max_seq_len = needed_by_mlp_max_seq_len
        # 对文本padding部分做全零初始化
        # self.embedding.weight.data[1]，即stoi['PAD']
        self.embedding.weight.data[1] = torch.zeros(self.embedding_dim)

        # 一个中间隐藏层
        self.one_hidden_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 所有中间隐藏层 - 1
        self.all_hidden_layer = nn.Sequential(self.one_hidden_layer)
        # 所有中间隐藏层 - 2
        for i in range(self.num_hidden_layers - 2):
            self.all_hidden_layer.add_module(str(i + 1), self.one_hidden_layer)

        # 隐藏层1层/n层
        if self.num_hidden_layers > 1:
            self.mlp = nn.Sequential(
                nn.Linear(self.max_seq_len * self.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                self.all_hidden_layer,
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.max_seq_len * self.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.output_dim),
            )

    def forward(self, text, _, text_lengths):
        # text [batch_size, seq_len]

        embedded = self.embedding(text).float()

        # embedded [batch_size, seq_len, emb_dim]

        embedded_ = embedded.view(embedded.size(0), -1)

        # embedded_ [batch_size, seq_len * emb_dim]

        out = self.mlp(embedded_)

        # [batch_size, output_dim]

        return out


class Bert(BaseModel):
    """
    Bert
    """

    def __init__(self, bert_path, num_classes, word_embedding, trained=True):
        """

        :param bert_path: bert预训练模型路径
        :param num_classes: 分类数
        :param word_embedding: None
        :param trained: 是否训练bert参数
        """

        super(Bert, self).__init__()

        # 从bert预训练模型文件，加载BertModel
        self.bert = BertModel.from_pretrained(bert_path)

        # 是否对bert进行训练
        """
        即是否对可训练参数求梯度
        
        原注释————不对bert进行训练
        """
        for param in self.bert.parameters():
            param.requires_grad = trained

        # 线性映射，bert的隐藏层维度，到，分类数
        self.fc = nn.Linear(self.bert.config.to_dict()['hidden_size'], num_classes)

    def forward(self, context, bert_masks, seq_lens):
        """
        :param context: 输入的句子序列

        :param bert_masks:
        构造的attention可视域的attention_mask，
        mask掉该位置后面的部分是为了保证模型不提前知道正确输出，
        因为那是要预测的呀！

        :param seq_lens: 句子长度
        """

        # context [batch_size, sen_len]

        # context传入bert模型，bert_masks标识要预测的部分
        _, cls = self.bert(context, attention_mask=bert_masks)
        # _ [batch_size, sen_len, H=768]
        # cls [batch_size, H=768]

        # 直接用cls预测
        out = self.fc(cls)
        # cls [hidden_size, num_classes]

        return out


class DRBert(nn.Module):
    """
    Bert+GRU
    """

    def __init__(self, rnn_type, bert_path, hidden_dim, n_layers, bidirectional, batch_first, word_embedding,
                 dropout, num_classes, trained):
        """

        :param rnn_type:
        :param bert_path:
        :param hidden_dim:
        :param n_layers:
        # :param bidirectional: 这里是True
        :param batch_first: 这里是True，配置json文件里写了
        :param word_embedding:
        :param dropout:
        :param num_classes:
        :param trained:
        """
        super(DRBert, self).__init__()
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = trained

        self.rnn_type = rnn_type.lower()
        # 可选的rnn类型
        if rnn_type == 'lstm':
            # 输入维度为bert的隐藏层维度
            self.rnn = nn.LSTM(self.bert.config.to_dict()['hidden_size'],
                               hidden_size=hidden_dim,
                               num_layers=n_layers,
                               bidirectional=bidirectional,
                               batch_first=batch_first,
                               dropout=dropout)
        elif rnn_type == 'gru':
            # 输入维度为bert的隐藏层维度
            self.rnn = nn.GRU(self.bert.config.to_dict()['hidden_size'],
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                            #   bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        else:
            self.rnn = nn.RNN(self.bert.config.to_dict()['hidden_size'],
                              hidden_size=hidden_dim,
                              num_layers=n_layers,
                              bidirectional=bidirectional,
                              batch_first=batch_first,
                              dropout=dropout)
        # 配置 bidirectional 为单向，这里的hidden_dim就是bert的隐藏层维度
        self.w = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

        # 配置 bidirectional 为双向，这里的hidden_dim就是bert的隐藏层维度
        # self.w = nn.Parameter(torch.randn(hidden_dim * 2), requires_grad=True)
        # self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, text, aspect, bert_masks, seq_lens, asp_len):
        # text shape: [batch_size, seq_len]
        bert_sentence, bert_cls = self.bert(text, attention_mask=bert_masks)
        # aspect shape: [batch_size, asp_len]
        bert_aspect, bert_cls_asp = self.bert(aspect, attention_mask=bert_masks)
        # bert_sentence shape: [batch_size, sen_len, H=768]
        # bert_cls shape: [batch_size, H=768] # cls 不用于预测
        bert_sentence, sorted_seq_lengths, desorted_indices = prepare_pack_padded_sequence(bert_sentence, seq_lens)
        bert_aspect = prepare_pack_padded_sequence_asp(bert_aspect, asp_len)
        # F function(i.e., dynamic re-weighting function)
        bert_con = F(bert_sentence, bert_aspect)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(bert_con, sorted_seq_lengths,
                                                            batch_first=self.batch_first)
        if self.rnn_type in ['rnn', 'gru']:
            packed_output, hidden = self.rnn(packed_embedded)
        else:
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # output = [seq_len, batch_size, hidden_dim * bidirectional]
        # hidden [n_layers * num_direction, batch_size, hidden_dim]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first)
        # output = [batch_size, seq_len, hidden_dim * bidirectional]
        output = output[desorted_indices]

        # 添加: 注意力机制
        m = self.tanh(output)
        score = torch.matmul(m, self.w)
        alpha = F.softmax(score, dim=0).unsqueeze(-1)
        output_attention = output * alpha
        output_attention = torch.sum(output_attention, dim=1)

        # hidden [n_layers * num_direction, batch_size, hidden_dim]
        hidden = torch.reshape(hidden, [output.shape[0], -1, output.shape[2]])
        hidden = torch.mean(hidden, dim=1)
        # hidden [batch_size, hidden_dim * bidirectional]
        output = torch.sum(output, dim=1)
        # output [batch_size, hidden_dim * bidirectional]
        fc_input = self.dropout(output + output_attention + hidden)
        # fc_input [batch_size, hidden_dim * bidirectional]
        out = self.fc(fc_input)
        # out [batch_size, num_classes]

        return out
