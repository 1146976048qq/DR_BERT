import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
import logging
from pytorch_pretrained_bert import BertModel, BertForSequenceClassification

from deepqa_models.transformer.transformer import ScaledDotProductAttention, clones, MultiHeadAttention, LayerNorm

class BertForAspect(BertForSequenceClassification):
    """
    Bert For aspect Task
    """
    def __init__(self, config, params):
        super(BertForAspect, self).__init__(config, params['n_labels'])
        self.n_labels = params['n_labels']
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.n_labels)
        self.apply(self.init_bert_weights)
        self.multiheads = MultiHeadAttention(config.hidden_size, params['heads'], keep_prob=params['keep_prob'])

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        Args:
          input_ids: (batch, seq_len), word index of text, start with [CLS] and end with [SEP] token ids
          token_type_ids: (batch, seq_len), values from [0,1], indicates whether it's from sentence A(0) or B(1)
          attention_mask: (batch, seq_len), mask for input text, values from [0,1], 1 means word is padded
          labels: (batch), y 
        """
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        logging.debug('bert for aspect: sequence_output {}'.format(sequence_output.shape))    

        pooled_output = self.dropout(sequence_output)
        pooled_output = self.multiheads(pooled_output)
        logging.debug('bert for aspect: multihead pooled_output shape {}'.format(pooled_output.shape))
        pooled_output = pooled_output[:,0,:]
        logging.debug('bert for aspect: pooled_out {}'.format(pooled_output.shape))
        
        # final prediction layer
        self.logits = self.classifier(pooled_output)
        logging.debug('bert for aspect: logits {}'.format(self.logits.shape))
        return self.logits

        # if self.n_labels > 2:
        #   self.prediction_result = F.softmax(self.logits, dim=-1)
        # else:
        #   self.prediction_result = F.sigmoid(self.logits)
        # logging.info('bert for aspect: prediction_result {}'.format(self.prediction_result.shape))

        # if labels is not None:
        #     loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        #     self.loss = loss_fn(self.logits.view(-1, self.n_labels), labels.view(-1))
        #     return self.loss
        # else:
        #     return self.logits

    # def get_loss(self):
    #     return self.loss
    
    # def get_prediction_result(self):
    #     return self.prediction_result
    
    # def get_logits(self):
    #     return self.logits
