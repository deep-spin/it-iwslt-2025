from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import AutoConfig


logger = logging.get_logger(__name__)


class SpeechLMConfig(PretrainedConfig):

    model_type = "speech_lm"
    sub_configs = {"encoder": AutoConfig, "decoder": AutoConfig}
    is_composition = True
    is_encoder_decoder = False
    add_pre_adapter: bool = False
    num_pre_adapter_layes: int = 3

    adapter_type: str = "linear"
    num_latents: int = 64
    has_no_defaults_at_init = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "encoder" not in kwargs or "decoder" not in kwargs:
            raise ValueError(
                f"A configuraton of type {self.model_type} cannot be instantiated because not both `encoder` and"
                f" `decoder` sub-configurations are passed, but only {kwargs}"
            )

        encoder_config = kwargs.pop("encoder")
        decoder_config = kwargs.pop("decoder")
        encoder_model_type = encoder_config.pop("model_type")
        decoder_model_type = decoder_config.pop("model_type")

        self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        self.decoder_start_token_id = self.decoder.bos_token_id
        self.loss_type = "ForCausalLMLoss"

    @property
    def vocab_size(self):
        return self.decoder.vocab_size

    @classmethod
    def from_encoder_decoder_configs(
        cls,
        encoder_config: PretrainedConfig,
        decoder_config: PretrainedConfig,
        **kwargs,
    ) -> PretrainedConfig:
        return cls(
            encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs
        )


__all__ = ["SpeechLMConfig"]
