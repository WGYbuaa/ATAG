import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertConfig, BertForPreTraining
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from keras.preprocessing.sequence import pad_sequences
from pytorch_transformers import AdamW, WarmupLinearSchedule

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 13


class D_Bert(nn.Module):
    def __init__(self):
        super(D_Bert, self).__init__()

        self.sentence_bert = BertModel.from_pretrained(MODEL_NAME)  # BERTtf
        self.sentence_dropout = nn.Dropout(.1)  # dropout
        self.paragraph_bert = BertModel.from_pretrained(MODEL_NAME)  # BERTtc
        self.paragraph_dropout = nn.Dropout(.1)
        self.classifier = nn.Linear(768, NUM_LABELS)  # fully connected layer

    def forward(self, sentence_tokens, sentence_masks, paragraph_tokens, paragraph_masks, tags = None):
        sentence_output, _ = self.sentence_bert(sentence_tokens, None, sentence_masks, output_all_encoded_layers=False)
        sentence_output = self.sentence_dropout(sentence_output)

        paragraph_output, _ = self.paragraph_bert(paragraph_tokens, None, paragraph_masks,
                                                  output_all_encoded_layers=False)
        paragraph_output = self.paragraph_dropout(paragraph_output)

        out = torch.cat((sentence_output, paragraph_output), 1)
        logits = self.classifier(out)

        return logits


