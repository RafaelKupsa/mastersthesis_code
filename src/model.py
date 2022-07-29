import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from transformers import XLMRobertaModel, XLMRobertaForMaskedLM
from transformers.utils.generic import ModelOutput
from transformers.models.roberta.modeling_roberta import RobertaLMHead

from .config import XLMRobertaForTLMandALPConfig, XLMRobertaForALPConfig

XLMRobertaForTLM = XLMRobertaForMaskedLM


class LPHead(nn.Module):
    """
    Link Prediction head consisting of two GATConv layers
    """
    def __init__(self, config):
        """
        Initializes the LPHead with the given config
        :param config: instance of XLMRobertaForALPConfig or XLMRobertaForTLMandALPConfig
        """
        super(LPHead, self).__init__()

        self.conv1 = GATConv(config.hidden_size, config.alp_hidden_size, heads=config.alp_num_heads, dropout=config.alp_dropout)
        self.conv2 = GATConv(config.alp_hidden_size * 2, config.alp_hidden_size, heads=1, dropout=config.alp_dropout)

    def forward(self, input_ids, edge_index, **kwargs):
        """
        forward pass of the LPHead
        :param input_ids: embedded nodes of the graph
        :param edge_index: edges to predict
        :param kwargs:
        :return: prediction logits for every edge in edge_index
        """
        out = []
        for i in range(input_ids.size(0)):
            conv_out = self.conv1(input_ids[i], edge_index[i])
            conv_out = F.elu(conv_out)
            conv_out = self.conv2(conv_out, edge_index[i])
            link_logits = (conv_out[edge_index[i][0]] * conv_out[edge_index[i][1]]).sum(dim=-1)
            out.append(link_logits)

        return torch.sigmoid(torch.stack(out))


class XLMRobertaForALP(XLMRobertaModel):
    """
    XLMRoberta for ALP (Alignment Link Prediction)
    """
    config_class = XLMRobertaForALPConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        """
        Initializes the model with the given config
        :param config: instance of XLMRobertaForALPConfig
        """
        super().__init__(config)

        self.xlmr = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.alp_head = LPHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids = None,
        attention_mask  = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        edge_index = None,
        labels = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        """
        forward pass of the model
        :return generic ModelOutput with loss and logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.xlmr(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.alp_head(sequence_output, edge_index)

        link_prediction_loss = None
        if labels is not None:
            loss_fct = torch.nn.BCELoss()
            link_prediction_loss = loss_fct(prediction_scores, labels)

        return ModelOutput(
            loss=link_prediction_loss,
            logits=prediction_scores
        )


class XLMRobertaForTLMandALP(XLMRobertaModel):
    """
    XLMRoberta for simultanrous TLM (Translation Language Modeling) and ALP (Alignment Link Prediction)
    """
    config_class = XLMRobertaForTLMandALPConfig
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        """
        Initializes the model with the given config
        :param config: instance of XLMRobertaForTLMandALPConfig
        """
        super().__init__(config)

        self.xlmr = XLMRobertaModel.from_pretrained("xlm-roberta-base")

        self.tlm_head = RobertaLMHead(config)
        self.alp_head = LPHead(config)

    def forward(
            self,
            input_ids = None,
            attention_mask = None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            edge_index=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        """
        forward pass of the model
        :return generic ModelOutput with loss and logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.xlmr(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        if edge_index is None:
            prediction_scores = self.tlm_head(sequence_output).view(-1, self.config.vocab_size)
            labels = labels.view(-1)
        else:
            prediction_scores = self.alp_head(sequence_output, edge_index)

        prediction_loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss() if edge_index is None else torch.nn.BCELoss()
            prediction_loss = loss_fct(prediction_scores, labels)

        return ModelOutput(
            loss=prediction_loss,
            logits=prediction_scores
        )