import torch.nn as nn
from transformers.modeling_outputs import Seq2SeqLMOutput


class ABSAPrefixForConditionalGeneration(nn.Module):
    def __init__(self, opt, t5_encoder, t5_model):
        super(ABSAPrefixForConditionalGeneration, self).__init__()
        self.opt = opt
        self.t5_encoder = t5_encoder
        self.t5_model = t5_model

    # 待确定

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,  # absa_labels？
            is_labels=None,
            dataset=None,
            decoder_attention_mask=None,
            return_dict=None
    ):
        outputs = self.t5_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=return_dict
        )
        loss = outputs.loss if labels is not None else None
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        decoder_hidden_states = outputs.decoder_hidden_states
        decoder_attentions = outputs.decoder_attentions
        cross_attentions = outputs.cross_attentions
        encoder_last_hidden_state = outputs.encoder_last_hidden_state

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            decoder_hidden_states=decoder_hidden_states,
            decoder_attentions=decoder_attentions,
            cross_attentions=cross_attentions,
            encoder_last_hidden_state=encoder_last_hidden_state
        )


class IsABSAModel(nn.Module):
    def __init__(self, absa_prefix_model, t5_model):
        super(IsABSAModel, self).__init__()
        self.absa_prefix_model = absa_prefix_model
        self.t5 = t5_model
        self.train_task = None

    def forward(self, absa_prefix_inputs, train_task=None):
        self.train_task = train_task
        absa_prefix_output = self.absa_prefix_model(**absa_prefix_inputs)
        return absa_prefix_output

    def predict(self, **kwargs):
        # print(kwargs)
        if self.train_task == "is":
            is_prefix_inputs = {
                "input_ids": kwargs["input_ids_bert"],
            }
            is_prefix_output = self.is_prefix_model(**is_prefix_inputs)
            ans = {"pred": is_prefix_output}
        else:
            out = self.t5.generate(
                input_ids=kwargs["input_ids_t5"],
                attention_mask=kwargs["attention_mask_t5"],
                max_length=kwargs["max_length"],
                num_beams=kwargs["num_beams"],
                output_hidden_states=True,
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True,
                **{'constraint_decoding': kwargs['constraint_decoding'],
                   'is_labels': kwargs['is_labels'], "dataset": kwargs['dataset']},
            )
            # ** {'next_ids': kwargs['next_ids'], 'constraint_decoding': kwargs['constraint_decoding']}

            ans = {"pred": out.sequences}
        return ans
