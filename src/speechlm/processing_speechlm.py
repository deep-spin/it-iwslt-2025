"""
Speech processor class for Wav2Vec2-BERT
"""

import warnings
from typing import List, Optional, Union, Dict

from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import AudioInput, PreTokenizedInput, TextInput
from ..seamless_m4t.feature_extraction_seamless_m4t import SeamlessM4TFeatureExtractor
from ..wav2vec2.tokenization_wav2vec2 import Wav2Vec2CTCTokenizer
from ..llama.tokenization_llama import LlamaTokenizer
from ... import AutoTokenizer
from tokenizers import AddedToken
import torch
import itertools

import pdb
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SpeechLMProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


def _have_same_length(items: list):
    return all(len(item) == len(items[0]) for item in items)


def _make_tensor(x: dict, device, dtype):
    return {k: torch.tensor(v, device=device, dtype=dtype) for k, v in x.items()}


class SpeechLMProcessor(ProcessorMixin):
    r"""
    Constructs a Wav2Vec2-BERT processor which wraps a Wav2Vec2-BERT feature extractor and a Wav2Vec2 CTC tokenizer into a single
    processor.

    [`Wav2Vec2Processor`] offers all the functionalities of [`SeamlessM4TFeatureExtractor`] and [`PreTrainedTokenizer`].
    See the docstring of [`~Wav2Vec2Processor.__call__`] and [`~Wav2Vec2Processor.decode`] for more information.

    Args:
        feature_extractor (`SeamlessM4TFeatureExtractor`):
            An instance of [`SeamlessM4TFeatureExtractor`]. The feature extractor is a required input.
        tokenizer ([`PreTrainedTokenizer`]):
            An instance of [`PreTrainedTokenizer`]. The tokenizer is a required input.
    """

    feature_extractor_class = "SeamlessM4TFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    lang2token = {
        # Official EU languages
        "bg": "<|bg|>",  # Bulgarian
        "hr": "<|hr|>",  # Croatian
        "cs": "<|cs|>",  # Czech
        "da": "<|da|>",  # Danish
        "nl": "<|nl|>",  # Dutch
        "en": "<|en|>",  # English
        "et": "<|et|>",  # Estonian
        "fi": "<|fi|>",  # Finnish
        "fr": "<|fr|>",  # French
        "de": "<|de|>",  # German
        "el": "<|el|>",  # Greek
        "hu": "<|hu|>",  # Hungarian
        "ga": "<|ga|>",  # Irish
        "it": "<|it|>",  # Italian
        "lv": "<|lv|>",  # Latvian
        "lt": "<|lt|>",  # Lithuanian
        "mt": "<|mt|>",  # Maltese
        "pl": "<|pl|>",  # Polish
        "pt": "<|pt|>",  # Portuguese
        "ro": "<|ro|>",  # Romanian
        "sk": "<|sk|>",  # Slovak
        "sl": "<|sl|>",  # Slovene
        "es": "<|es|>",  # Spanish
        "sv": "<|sv|>",  # Swedish
        "sq": "<|sq|>",  # Albanian
        # Other languages
        "ast": "<|ast|>",  # Asturian
        "eu": "<|eu|>",  # Basque
        "br": "<|br|>",  # Breton
        "ca": "<|ca|>",  # Catalan
        "fy": "<|fy|>",  # Frisian
        "gl": "<|gl|>",  # Galician
        "oc": "<|oc|>",  # Occitan
        "rm": "<|rm|>",  # Romansh (Vallader, Sursilv)
        "sc": "<|sc|>",  # Sardinian
        "hsb": "<|hsb|>",  # Sorbian
        "cy": "<|cy|>",  # Welsh
        "zh": "<|zh|>",  # Chinese
    }

    task2token = {
        "transcribe": "<|transcribe|>",
        "translate": "<|translate|>",
        "summarize": "<|summarize|>",
        "sqa": "<|reply|>",
    }

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls, audio_model_name_or_path, text_model_name_or_path, **kwargs
    ):
        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            audio_model_name_or_path, **kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(text_model_name_or_path, **kwargs)

        additional_tokens = [
            *list(cls.task2token.values()),
            *list(cls.lang2token.values()),
        ]

        tokenizer.add_tokens(additional_tokens, special_tokens=True)
        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def _build_preamble_block(
        self, target_lang, target_task, text_preamble: str | None = None
    ):
        if text_preamble is not None:
            preamble_block = [
                f"{self.lang2token[tl]}{self.task2token[tt]}{tp}"
                for tl, tt, tp in zip(target_lang, target_task, text_preamble)
            ]
        else:
            preamble_block = [
                f"{self.lang2token[tl]}{self.task2token[tt]}"
                for tl, tt in zip(target_lang, target_task)
            ]

        # We pad becasue because the text preamble might contain some signal
        # for certain items in the batch or not for others.
        preamble_inputs = self.tokenizer(
            preamble_block,
            padding="longest",
            padding_side="left",
            return_attention_mask=True,
            add_special_tokens=False,
        )
        return preamble_inputs

    def __call__(
        self,
        audio: AudioInput,
        task: str | list[str],
        target_lang: str | list[str],
        text: str | list[str] | TextInput | PreTokenizedInput = None,
        text_preamble: str | list[str] = None,
        return_labels: bool = False,
        **kwargs,
    ):
        """
        TODO: update docstring
        """
        if return_labels and text is None:
            raise ValueError("You need to specify a `text` input to return labels.")

        if audio is None and text is None:
            raise ValueError(
                "You need to specify either an `audio` or `text` input to process."
            )

        if "return_tensors" in kwargs and kwargs["return_tensors"] != "pt":
            warnings.warn(
                "We currently support `return_tensors='pt'`. Setting it to `pt` for now.",
            )
        kwargs["return_tensors"] = "pt"

        output_kwargs = self._merge_kwargs(
            SpeechLMProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        # 1. build audio component of the input
        # TODO: for now, audio is mandatory
        audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
        items_count = audio_inputs["input_features"].shape[0]

        # 2. audio is followed by a block containing ids for task, target_lang, and optionally text preamble
        target_lang = (
            target_lang
            if isinstance(target_lang, list)
            else [target_lang] * items_count
        )
        task = task if isinstance(task, list) else [task] * items_count
        len_to_check = [target_lang, task]
        if text_preamble is not None:
            text_preamble = (
                text_preamble
                if isinstance(text_preamble, list)
                else [text_preamble] * items_count
            )
            len_to_check.append(text_preamble)
        if not _have_same_length(len_to_check):
            raise ValueError(
                "`lang`, `task`, and `text_preamble` must have the same length"
            )

        preamble_inputs = self._build_preamble_block(target_lang, task, text_preamble)

        # 3. encode text if needed
        text_inputs = preamble_inputs
        if text is not None:
            if not isinstance(text, list):
                text = [text]
            if not _have_same_length([text, target_lang]):
                raise ValueError("`text` and `lang` must have the same length")

            # TODO: we should truncate the input text!

            tokenized_text = self.tokenizer(
                text,
                padding=False,
                truncation=False,
                add_special_tokens=False,
                return_attention_mask=True,
            )
            tokenized_text["input_ids"] = [
                tt + [self.tokenizer.eos_token_id] for tt in tokenized_text["input_ids"]
            ]
            tokenized_text["attention_mask"] = [
                tt + [1] for tt in tokenized_text["attention_mask"]
            ]

            text_inputs = {
                "input_ids": [
                    pi + tt
                    for pi, tt in zip(
                        preamble_inputs["input_ids"], tokenized_text["input_ids"]
                    )
                ],
                "attention_mask": [
                    pi + tt
                    for pi, tt in zip(
                        preamble_inputs["attention_mask"],
                        tokenized_text["attention_mask"],
                    )
                ],
            }

            # text inputs should be in this moment: (tokenized)
            # <|en|><|transcribe|> 5 I went to the store</s>
            # <|en|><|transcribe|> 8 I went to the store with my daughter.</s>
            # here we want to pad everything
            text_inputs = self.tokenizer.pad(
                text_inputs,
                padding_side="left",
                padding="longest",
                return_tensors="pt",
            )
            # here we should be like this
            # <pad><pad><pad><pad><|en|><|transcribe|> 5 I went to the store</s>
            # <|en|><|transcribe|> 8 I went to the store with my daughter.</s>
        else:
            # if we don't have text to learn we are generating, hence create tensors out of the preamble
            text_inputs = _make_tensor(
                text_inputs,
                device=audio_inputs["input_features"].device,
                dtype=torch.long,
            )

        output_dict = {
            **{f"audio_{k}": v for k, v in audio_inputs.items()},
            **text_inputs,
        }
        if return_labels:
            output_dict["labels"] = self.tokenizer.pad(
                tokenized_text,
                padding_side="left",
                padding="longest",
                return_tensors="pt",
            )["input_ids"]

        return output_dict

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)


# __all__ = ["Wav2Vec2BertProcessor"]
