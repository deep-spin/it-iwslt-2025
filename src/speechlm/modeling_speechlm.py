# coding=utf-8
"""Speech LM architecture"""

from typing import Optional, Tuple, Union
import pdb
import torch
from torch import nn
import torch.nn.functional as F

from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel, AutoModelForCausalLM
from .configuration_speechlm import SpeechLMConfig

from ..wav2vec2_bert.modeling_wav2vec2_bert import Wav2Vec2BertAdapterLayer


logger = logging.get_logger(__name__)


class SpeechLMPreTrainedModel(PreTrainedModel, GenerationMixin):
    base_class_prefix = "model"
    _skip_keys_device_placement = ["past_key_values"]
    _no_split_modules = [
        "LlamaDecoderLayer",
        "SpeechLMPreAdapter",
        "Wav2Vec2BertAdapterLayer",
        "Wav2Vec2BertEncoderLayer",
    ]


class SpeechLMPreAdapter(nn.Module):
    def __init__(self, config):

        super().__init__()
        # feature dim might need to be down-projected
        self.proj = nn.Linear(
            config.encoder.feature_projection_input_dim,
            config.encoder.output_hidden_size,
        )
        self.proj_layer_norm = nn.LayerNorm(
            config.encoder.output_hidden_size, eps=config.encoder.layer_norm_eps
        )
        self.layers = nn.ModuleList(
            Wav2Vec2BertAdapterLayer(config.encoder)
            for _ in range(config.num_pre_adapter_layers)
        )
        self.layerdrop = config.encoder.layerdrop

        self.kernel_size = config.encoder.adapter_kernel_size
        self.stride = config.encoder.adapter_stride
        self.out_proj = nn.Linear(
            config.encoder.output_hidden_size,
            config.encoder.feature_projection_input_dim,
        )

    def _compute_sub_sample_lengths_from_attention_mask(self, seq_lens):
        if seq_lens is None:
            return seq_lens
        pad = self.kernel_size // 2
        seq_lens = ((seq_lens + 2 * pad - self.kernel_size) / self.stride) + 1
        return seq_lens.floor()

    def forward(self, hidden_states, attention_mask=None):
        # down project hidden_states if necessary
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        sub_sampled_lengths = None
        if attention_mask is not None:
            sub_sampled_lengths = (
                attention_mask.size(1) - (1 - attention_mask.int()).sum(1)
            ).to(hidden_states.device)

        for layer in self.layers:
            layerdrop_prob = torch.rand([])
            sub_sampled_lengths = self._compute_sub_sample_lengths_from_attention_mask(
                sub_sampled_lengths
            )
            if not self.training or (layerdrop_prob > self.layerdrop):
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    sub_sampled_lengths=sub_sampled_lengths,
                )

        hidden_states = self.out_proj(hidden_states)
        return hidden_states


# @add_start_docstrings(SPEECH_ENCODER_DECODER_START_DOCSTRING)
class SpeechLMForConditionalGeneration(SpeechLMPreTrainedModel):
    r"""
    [`SpeechEncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture with
    one of the base model classes of the library as encoder and another one as decoder when created with the
    :meth*~transformers.AutoModel.from_pretrained* class method for the encoder and
    :meth*~transformers.AutoModelForCausalLM.from_pretrained* class method for the decoder.
    """

    config_class = SpeechLMConfig
    base_model_prefix = "speech_lm"
    main_input_name = "input_ids"

    supports_gradient_checkpointing = True
    _supports_param_buffer_assignment = False
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    # TODO: this gets ignored by the trainer, maybe it goes in the config class
    loss_type: str = "ForCausalLM"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        if config is None and (encoder is None or decoder is None):
            raise ValueError(
                "Either a configuration or an encoder and a decoder has to be provided."
            )
        if config is None:
            config = SpeechLMForConditionalGeneration.from_encoder_decoder_configs(
                encoder.config, decoder.config
            )
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(
                    f"Config: {config} has to be of type {self.config_class}"
                )

        self.loss_type = "ForCausalLM"

        # initialize with config
        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        super().__init__(config)

        if encoder is None:
            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:
            decoder = AutoModelForCausalLM.from_config(config.decoder)

        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.config.encoder._attn_implementation = (
            self.encoder.config._attn_implementation
        )
        self.config.decoder._attn_implementation = (
            self.decoder.config._attn_implementation
        )
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        # get encoder output hidden size
        self.encoder_output_dim = getattr(
            config.encoder,
            "output_hidden_size",
            config.encoder.hidden_size,
        )

        ####################################
        # MODALITY AND LENGTH ADAPTER
        # TODO: be back at this with better strategies
        ####################################
        if self.encoder_output_dim != self.decoder.config.hidden_size:
            logger.info("Adding encoder to decoder projection layer")
            self.enc_to_dec_proj = nn.Linear(
                self.encoder.config.hidden_size, self.decoder.config.hidden_size
            )

        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )

        if config.add_pre_adapter:
            self.pre_adapter = SpeechLMPreAdapter(self.config)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder of the speech encoder so
        that its parameters will not be updated during training.
        We freeze all parameters but the adapter.
        """
        for name, param in self.encoder.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False

    def freeze_decoder(self):
        """
        Calling this function will disable the gradient computation for the decoder so that its parameters will not be
        updated during training.
        """
        for param in self.decoder.parameters():
            param.requires_grad = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for SpeechEncoderDecoderModel. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False

        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs,
    ) -> PreTrainedModel:
        kwargs_encoder = {
            argument[len("encoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path,
                    **kwargs_encoder,
                    return_unused_kwargs=True,
                )

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(
                encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder
            )

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path,
                    **kwargs_decoder,
                    return_unused_kwargs=True,
                )

                kwargs_decoder["config"] = decoder_config

            decoder = AutoModelForCausalLM.from_pretrained(
                decoder_pretrained_model_name_or_path, **kwargs_decoder
            )

        # instantiate config with corresponding kwargs
        config = SpeechLMConfig.from_encoder_decoder_configs(
            encoder.config, decoder.config, **kwargs
        )

        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        return cls(encoder=encoder, decoder=decoder, config=config)

    # @add_start_docstrings_to_model_forward(SPEECH_ENCODER_DECODER_INPUTS_DOCSTRING)
    # @replace_return_docstrings(
    #     output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
    # )
    def forward(
        self,
        audio_input_features: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        audio_attention_mask: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor = None,
        # encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,  # no need for now
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # no need for now
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # input_values: Optional[torch.FloatTensor] = None,  # no need for now
        # input_features: Optional[torch.FloatTensor] = None,  # no need for now
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        kwargs_encoder = {
            argument: value
            for argument, value in kwargs.items()
            if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("decoder_")
        }
        if "num_items_in_batch" in kwargs_encoder:
            kwargs_decoder["num_items_in_batch"] = kwargs_encoder.pop(
                "num_items_in_batch", None
            )

        # we assume that if we are using cache then we are caching encoder_outputs
        if not use_cache or (use_cache and past_key_values is None):

            if self.config.add_pre_adapter:
                audio_input_features = self.pre_adapter(
                    audio_input_features, attention_mask=audio_attention_mask
                )
                audio_attention_mask = self.encoder._get_feature_vector_attention_mask(
                    audio_input_features.shape[1], audio_attention_mask
                )

            encoder_outputs = self.encoder(
                audio_input_features,
                attention_mask=audio_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
            encoder_hidden_states = encoder_outputs[0]

            ##########################
            # (Optional) ADAPTER
            ##########################
            # we project the encoder outputs if we haven't done it
            # within the adapter
            # TODO: the adapter part has to be improved / updated
            if hasattr(self, "enc_to_dec_proj"):
                encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

            # self.encoder_hidden_states = encoder_hidden_states
            if audio_attention_mask is not None:
                encoder_outputs_mask = self.encoder._get_feature_vector_attention_mask(
                    encoder_hidden_states.shape[1], audio_attention_mask
                )
            else:
                encoder_outputs_mask = torch.ones(
                    encoder_hidden_states.shape[:2],
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

            # if use_cache:
            # self.encoder_outputs_mask = encoder_outputs_mask

        # else:
        # encoder_hidden_states = self.encoder_hidden_states
        # store the encoder attention mask if we are using cache
        # elif use_cache and not hasattr(self, "audio_attention_mask"):
        # self.audio_attention_mask = audio_attention_mask

        # TODO: keeping it here as it might be useful in the future
        # compute correct encoder attention mask
        # if attention_mask is not None:
        #     encoder_attention_mask = self.encoder._get_feature_vector_attention_mask(
        #         encoder_hidden_states.shape[1], attention_mask
        #     )
        # else:
        #     encoder_attention_mask = None

        # extract input embeds from the decoder
        decoder_input_embs = self.decoder.get_input_embeddings()(input_ids)

        # If we are not using the cache, or it's the first pass with the cache on.
        # Hence, we need to build new inputs for the decoder
        if not use_cache or (use_cache and past_key_values is None):
            # prepend audio representations to the text input embeddings
            decoder_input_embs = torch.cat(
                [encoder_hidden_states, decoder_input_embs], dim=1
            )
            # encoder_outputs_mask = torch.ones(
            #     encoder_hidden_states.shape[:2],
            #     dtype=attention_mask.dtype,
            #     device=attention_mask.device,
            # )

            if attention_mask is not None:
                attention_mask = torch.cat(
                    [encoder_outputs_mask, attention_mask], dim=1
                )
        # else:
        #     if attention_mask is not None:
        #         attention_mask = torch.cat(
        #             [self.audio_attention_mask, attention_mask], dim=1
        #         )

        if logits_to_keep == 0:
            logits_to_keep = (
                labels.shape[1] if labels is not None else input_ids.shape[1]
            )

        # pdb.set_trace()

        decoder_outputs = self.decoder(
            inputs_embeds=decoder_input_embs,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            logits_to_keep=logits_to_keep,  # TODO: does this work only for llama decoders?
            **kwargs_decoder,
        )

        logits = decoder_outputs.logits

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=decoder_outputs.logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                ignore_index=self.config.decoder.pad_token_id,
                **kwargs,
            )

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resizing the embedding layers via the SpeechEncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped decoder object (model.decoder.resize_token_embeddings(...))"
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past_key_values, beam_idx)

    # GA: tweak to clean the encoder attention mask for each new generation
    # Since that attention mask is not handled natively from HF's internal loop
    def generate(self, *args, **kwargs):
        if hasattr(self, "audio_attention_mask"):
            del self.audio_attention_mask
        return super().generate(*args, **kwargs)

    # TODO: tweak to use the model with HF's generate.
    # We should not need it anymore starting from transformers 4.50, but let's see
    def can_generate(self):
        return True


__all__ = ["SpeechLMPreTrainedModel", "SpeechLMForConditionalGeneration"]
