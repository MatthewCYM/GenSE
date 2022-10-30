import copy
import warnings
import logging
import torch
import torch.nn as nn
from transformers.models.t5.modeling_t5 import (
    T5ForConditionalGeneration,
    T5Stack,
    get_device_map,
    assert_device_map,
    Seq2SeqLMOutput,
    BaseModelOutput,
)
from modeling_utils import (
    Similarity,
    Pooler,
    MLPLayer,
    all_gather
)
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import torch.distributed as dist


logger = logging.getLogger(__name__)


class T5ForSentenceEmbedding(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.model_dim = config.d_model
        self.model_type = self.model_args.model_type
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        if self.model_type == 'seq2seq':
            decoder_config = copy.deepcopy(config)
            decoder_config.is_decoder = True
            decoder_config.is_encoder_decoder = False
            decoder_config.num_layers = config.num_decoder_layers
            self.decoder = T5Stack(decoder_config, self.shared)

            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize contrastive loss components
        self.pooler_type = self.model_args.pooler_type
        self.pooler = Pooler(self.model_args.pooler_type)
        if self.model_args.pooler_type == 'cls':
            self.mlp = MLPLayer(config)
        self.sim = Similarity(temp=self.model_args.temp)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        if self.model_type == 'seq2seq':
            self.decoder.parallelize(self.device_map)
        self.pooler = self.pooler.to(self.encoder.last_device)
        if self.model_args.pooler_type == 'cls':
            self.mlp = self.mlp.to(self.encoder.last_device)
        self.sim = self.sim.to(self.encoder.last_device)
        if self.model_type == 'seq2seq':
            self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        if self.model_type == 'seq2seq':
            self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        if self.model_type == 'seq2seq':
            self.decoder = self.decoder.to("cpu")
        self.pooler = self.pooler.to("cpu")
        if self.model_args.pooler_type == 'cls':
            self.mlp = self.mlp.to("cpu")
        self.sim = self.sim.to("cpu")
        if self.model_type == 'seq2seq':
            self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def sentemb_forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=True if self.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )
        if self.model_type == 'encoder':
            pooler_output = self.pooler(attention_mask, outputs)
            if self.pooler_type == "cls" and not self.model_args.mlp_only_train:
                pooler_output = self.mlp(pooler_output)
            if not return_dict:
                return (outputs[0], pooler_output) + outputs[2:]

            return BaseModelOutputWithPoolingAndCrossAttentions(
                pooler_output=pooler_output,
                last_hidden_state=outputs.last_hidden_state,
                hidden_states=outputs.hidden_states,
            )
        decoder_start_token_id = self._get_decoder_start_token_id()
        decoder_input_ids = torch.full([input_ids.shape[0], 1], decoder_start_token_id).to(self.device)
        hidden_states = outputs.last_hidden_state
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )
        pooler_output = self.pooler(None, decoder_outputs)
        if self.pooler_type == "cls" and not self.model_args.mlp_only_train:
            pooler_output = self.mlp(pooler_output)
        if not return_dict:
            return (outputs[0], pooler_output) + outputs[2:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=decoder_outputs.last_hidden_state,
            hidden_states=decoder_outputs.hidden_states,
        )

    def compute_contrastive_loss(
        self,
        encoder_outputs,
        attention_mask,
        batch_size,
        num_sent,
    ):
        # Pooling
        pooler_output = self.pooler(attention_mask, encoder_outputs)
        pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))  # (bs, num_sent, hidden)

        # If using "cls", we add an extra MLP layer
        # (same as BERT's original implementation) over the representation.
        if self.pooler_type == "cls":
            pooler_output = self.mlp(pooler_output)

        # Separate representation
        z1, z2 = pooler_output[:, 0], pooler_output[:, 1]

        # Hard negative
        if num_sent == 3:
            z3 = pooler_output[:, 2]

        # Gather all embeddings if using distributed training
        if dist.is_initialized() and self.training:
            # Gather hard negative
            if num_sent >= 3:
                z3 = all_gather(z3)
            z1 = all_gather(z1)
            z2 = all_gather(z2)
        loss_fct = nn.CrossEntropyLoss()
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        # Hard negative
        if num_sent >= 3:
            z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

        labels = torch.arange(cos_sim.size(0)).long().to(self.device)

        # Calculate loss with hard negatives
        if num_sent == 3:
            # Note that weights are actually logits of weights
            z3_weight = self.model_args.hard_negative_weight
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(self.device)
            cos_sim = cos_sim + weights

        loss = loss_fct(cos_sim, labels)
        return loss

    def loss_forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        if encoder_outputs is None:
            batch_size, num_sent, seq_len = input_ids.shape
            assert len(input_ids.shape) == 3 and num_sent > 1
            input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
            attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent, len)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.view((-1, decoder_input_ids.size(-1)))
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=True if self.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
                return_dict=True,
            )
            hidden_states = encoder_outputs.last_hidden_state
            if self.model_type == 'encoder':
                contrastive_loss = self.compute_contrastive_loss(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    batch_size=batch_size,
                    num_sent=num_sent
                )
                return Seq2SeqLMOutput(
                    loss=contrastive_loss,
                    logits=None,
                    past_key_values=None,
                    decoder_hidden_states=None,
                    decoder_attentions=None,
                    cross_attentions=None,
                    encoder_last_hidden_state=None,
                    encoder_hidden_states=None,
                    encoder_attentions=None,
                )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
            hidden_states = encoder_outputs.last_hidden_state

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        contrastive_loss = self.compute_contrastive_loss(
            encoder_outputs=decoder_outputs,
            attention_mask=None,
            batch_size=batch_size,
            num_sent=num_sent
        )
        return Seq2SeqLMOutput(
            loss=contrastive_loss,
            logits=None,
            past_key_values=None,
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=None,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
    ):
        if sent_emb:
            return self.sentemb_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )
        else:
            return self.loss_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
