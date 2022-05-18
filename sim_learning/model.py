import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
import torch.nn.functional as F
from dataclasses import dataclass

from typing import Optional, Tuple

from transformers.activations import ACT2FN, gelu
from transformers.file_utils import ModelOutput
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel


# Copied from transformers.modeling_roberta.RobertaLMHead
class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x
    
    
# Copied from transformers.modeling_roberta.RobertaClassificationHead
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    
class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum()
        return b
    

class SimbertForPreTraining(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
            
        self.roberta = RobertaModel(config)
        if self.config.mlm_layer2:
            self.mlm_head2 = RobertaLMHead(config)
        if self.config.mlm_layer4:
            self.mlm_head4 = RobertaLMHead(config)
        if self.config.mlm_layer6:
            self.mlm_head6 = RobertaLMHead(config)
        if self.config.mlm_layer8:
            self.mlm_head8 = RobertaLMHead(config)
        if self.config.mlm_layer10:
            self.mlm_head10 = RobertaLMHead(config)
        if self.config.mlm_layer12:
            self.mlm_head12 = RobertaLMHead(config)
        self.classifier = RobertaClassificationHead(config)
        self.entropy_layer = RobertaClassificationHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        return_dict=None,
        antonym_ids=None,
        antonym_label=None,
        synonym_ids=None,
        synonym_label=None,
        **kwargs,
    ):
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        seq_len = input_ids.size()[1] // 2
        class_labels, mlm_labels = labels[:, 0], labels[:, 1:]
        a, b = input_ids[:, :seq_len], input_ids[:, seq_len:]
        input_ids = torch.cat((a, b), dim=0)
        a, b = attention_mask[:, :seq_len], attention_mask[:, seq_len:]
        attention_mask = torch.cat((a, b), dim=0)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            return_dict=return_dict,
            output_hidden_states=True
        )
        batch_size = input_ids.size(0) // 2
        sequence_output, pooled_output = outputs[:2]
        class_scores = self.classifier(sequence_output[:batch_size])
        hidden_states = outputs[2]

        if self.config.mlm_layer2:
            mlm_scores2 = self.mlm_head2(hidden_states[2][batch_size:])
        if self.config.mlm_layer4:
            mlm_scores4 = self.mlm_head4(hidden_states[4][batch_size:])
        if self.config.mlm_layer6:
            mlm_scores6 = self.mlm_head6(hidden_states[6][batch_size:])
        if self.config.mlm_layer8:
            mlm_scores8 = self.mlm_head8(hidden_states[8][batch_size:])
        if self.config.mlm_layer10:
            mlm_scores10 = self.mlm_head10(hidden_states[10][batch_size:])
        if self.config.mlm_layer12:
            mlm_scores12 = self.mlm_head12(hidden_states[12][batch_size:])
            
        if self.config.use_entropy_layer:
            entropy_layer = self.entropy_layer
        else:
            entropy_layer = self.classifier
        if self.config.entropy_layer2:
            entropy_scores2 = entropy_layer(hidden_states[2][batch_size:])
        if self.config.entropy_layer4:
            entropy_scores4 = entropy_layer(hidden_states[4][batch_size:])
        if self.config.entropy_layer6:
            entropy_scores6 = entropy_layer(hidden_states[6][batch_size:])
        if self.config.entropy_layer8:
            entropy_scores8 = entropy_layer(hidden_states[8][batch_size:])
        if self.config.entropy_layer10:
            entropy_scores10 = entropy_layer(hidden_states[10][batch_size:])
        if self.config.entropy_layer12:
            entropy_scores12 = entropy_layer(hidden_states[12][batch_size:])

        total_loss = None
        if labels is not None:
            loss_tok = CrossEntropyLoss()
            loss_h = HLoss()
            loss_class = CrossEntropyLoss()
            class_loss = loss_class(class_scores, class_labels)

            total_loss = class_loss
            if self.config.mlm_layer2:
                mlm_loss = loss_tok(mlm_scores2.view(-1, self.config.vocab_size), mlm_labels.reshape(-1))
                total_loss += mlm_loss
            if self.config.mlm_layer4:
                mlm_loss = loss_tok(mlm_scores4.view(-1, self.config.vocab_size), mlm_labels.reshape(-1))
                total_loss += mlm_loss
            if self.config.mlm_layer6:
                mlm_loss = loss_tok(mlm_scores6.view(-1, self.config.vocab_size), mlm_labels.reshape(-1))
                total_loss += mlm_loss
            if self.config.mlm_layer8:
                mlm_loss = loss_tok(mlm_scores8.view(-1, self.config.vocab_size), mlm_labels.reshape(-1))
                total_loss += mlm_loss
            if self.config.mlm_layer10:
                mlm_loss = loss_tok(mlm_scores10.view(-1, self.config.vocab_size), mlm_labels.reshape(-1))
                total_loss += mlm_loss
            if self.config.mlm_layer12:
                mlm_loss = loss_tok(mlm_scores12.view(-1, self.config.vocab_size), mlm_labels.reshape(-1))
                total_loss += mlm_loss
            
            if self.config.entropy_layer2:
                entropy_loss = loss_h(entropy_scores2)
                total_loss += entropy_loss
            if self.config.entropy_layer4:
                entropy_loss = loss_h(entropy_scores4)
                total_loss += entropy_loss
            if self.config.entropy_layer6:
                entropy_loss = loss_h(entropy_scores6)
                total_loss += entropy_loss
            if self.config.entropy_layer8:
                entropy_loss = loss_h(entropy_scores8)
                total_loss += entropy_loss
            if self.config.entropy_layer10:
                entropy_loss = loss_h(entropy_scores10)
                total_loss += entropy_loss
            if self.config.entropy_layer12:
                entropy_loss = loss_h(entropy_scores12)
                total_loss += entropy_loss
            #print(mlm_loss.item(), tec_loss.item(), sec_loss.item())
        if not return_dict:
            output = (class_scores,)
            return ((total_loss,) + output) if total_loss is not None else output
        return SimbertOutput(
            loss=total_loss,
            prediction_logits=class_scores,
        )


@dataclass
class SimbertOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None

        
from transformers import RobertaConfig

class SimConfig(RobertaConfig):
    mlm_layer6 = False
    mlm_layer4 = False
    mlm_layer2 = False
    mlm_layer8 = False
    mlm_layer10 = False
    mlm_layer12 = False
    entropy_layer6 = False
    entropy_layer4 = False
    entropy_layer2 = False
    entropy_layer8 = False
    entropy_layer10 = False
    entropy_layer12 = False
    use_entropy_layer = False
