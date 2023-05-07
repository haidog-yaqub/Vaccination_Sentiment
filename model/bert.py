import torch
from torch import nn


class BERT(nn.Module):

    def __init__(self, options_name = "bert-base-uncased", cls=2):
        super(BERT, self).__init__()
        self.encoder = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', options_name)
        fc_inputs = self.encoder.classifier.in_features
        self.encoder.classifier = nn.Linear(fc_inputs, cls)

    def forward(self, text, mask):
        x = self.encoder(text, token_type_ids=None, attention_mask=mask, labels=None)['logits']
        return x