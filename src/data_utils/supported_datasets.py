from typing import List
from datasets import (
    concatenate_datasets,
    load_dataset,
    DatasetDict,
    load_from_disk,
    Audio,
    Dataset,
    IterableDataset,
)
from tqdm import tqdm
import os
import random
import logging
import pandas as pd
from . import _local_dataset_dir
from sklearn.model_selection import train_test_split


# Setup logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def subsample_split(data, split: str | list[str], samples):
    if isinstance(split, str):
        split = [split]

    if samples is not None:
        for s in split:
            data[s] = data[s].select(
                random.sample(range(len(data[s])), min(samples, len(data[s])))
            )
    return data


class InfiniteValue:
    def __init__(self, value):
        self.value = value

    def __iter__(self):
        # This makes the object iterable
        return self

    def __next__(self):
        # Always returns the same value
        return self.value

    def __getitem__(self, index):
        # Makes the object subscriptable for any index
        # Ignores the index and always returns the value
        return self.value


def add_lang_task_columns(d: DatasetDict | Dataset, lang: str, task: str):
    """
    Adds a language column to a Dataset or DatasetDict.
    This function adds a column named "lang" to a dataset,
    where each element of the column has the value specified by the `lang` parameter.
    Args:
        d: Union[DatasetDict, Dataset]
            The dataset to modify. Can be either a Dataset or a DatasetDict.
        lang: str
            The language identifier to add as a column value for all examples.
    Returns:
        Union[DatasetDict, Dataset]: The modified dataset with the added 'lang' column.
            If input is a Dataset, returns a Dataset.
            If input is a DatasetDict, returns a DatasetDict with the column added to each split.
    Raises:
        ValueError: If the input is neither a Dataset nor a DatasetDict.
    Example:
        >>> from datasets import Dataset
        >>> ds = Dataset.from_dict({"text": ["Hello", "World"]})
        >>> ds_with_lang = add_lang_task_columns(ds, "en", "asr")
        >>> print(ds_with_lang["lang"])
        ['en', 'en']
    """

    def _add_lang(x):
        l_value = (
            InfiniteValue(lang) if isinstance(x, IterableDataset) else [lang] * len(x)
        )
        x = x.add_column("lang", l_value)
        t_value = (
            InfiniteValue(task) if isinstance(x, IterableDataset) else [task] * len(x)
        )
        return x.add_column("task", t_value)

    if isinstance(d, Dataset):
        return _add_lang(d)
    elif isinstance(d, DatasetDict) or isinstance(d, dict):
        return DatasetDict({k: _add_lang(v) for k, v in d.items()})
    else:
        raise ValueError(f"Unknown dataset type {type(d)}")


def add_length_seconds(
    dataset: Dataset, length_cachefile: str, dataset_workers: int = 1
):
    def _compute_len(batch):
        return {"length": [len(a["array"]) // a["sampling_rate"] for a in batch]}

    if "length" in dataset.column_names:
        logger.debug("Length column already exists, skipping length computation.")
        return dataset

    if os.path.exists(length_cachefile):
        logger.debug(f"Loading lengths from cache file: {length_cachefile}")
        df = pd.read_csv(length_cachefile)
        lengths_ds = Dataset.from_pandas(df)  # o_add = df["length"]

    else:
        lengths_ds = dataset.map(
            _compute_len,
            batched=True,
            num_proc=dataset_workers,
            batch_size=4000,
            input_columns=["audio"],
            remove_columns=dataset.column_names,
            keep_in_memory=True,
            desc="Computing length in seconds",
        )
        lengths_ds.to_csv(length_cachefile)

    logger.debug(f"Adding length column to dataset from file: {length_cachefile}")
    dataset = concatenate_datasets([dataset, lengths_ds], axis=1)
    return dataset


class BaseDataset:

    if_hf_dataset: bool = None
    hf_name: str = None
    hf_config: str | list[str] = None
    text_col: str = None
    remove_columns: list[str] = None
    lang: str = None
    task: str = None
    validation_split_name: str = "validation"

    @classmethod
    def get_dataset_dict(cls, splits: list[str], hf_config=None, **kwargs):
        hf_name = kwargs.get("hf_name", cls.hf_name)
        hf_config = hf_config if hf_config else cls.hf_config

        if _local_dataset_dir is not None:
            data = load_from_disk(os.path.join(_local_dataset_dir, cls.hf_name))
            # Pop all keys that are not in splits
            for key in list(data.keys()):
                if key not in splits:
                    data.pop(key)
        else:
            data = DatasetDict(
                {s: load_dataset(hf_name, hf_config, split=s, **kwargs) for s in splits}
            )

        # Add the length in seconds to each split
        found_splits = data.keys()
        for fs in found_splits:
            data[fs] = add_length_seconds(
                data[fs],
                os.path.join(
                    os.getenv("BASEDIR"),
                    f"length_{cls.nickname}_{cls.lang}_{fs}.csv",
                ),
            )

        # Add the lang and task metadata
        data = add_lang_task_columns(data, cls.lang, cls.task)

        # Subsample the validation set if requested
        data = subsample_split(
            data, cls.validation_split_name, kwargs.get("samples_validation", None)
        )

        # Remove unused columns
        data = data.remove_columns(cls.remove_columns)
        if cls.text_col != "text":
            data = data.rename_column("text", cls.text_col)

        return data

    @classmethod
    def download_and_save_arrow(cls, save_dir: str = None, num_proc: int = 1):
        if save_dir is None and _local_dataset_dir is None:
            raise ValueError(
                "Please provide a save_dir or set the LOCAL_DATASETS_DIR environment variable."
            )

        target_dir = os.path.join(save_dir or _local_dataset_dir, cls.nickname)
        logger.info(f"Downloading each language split in: {target_dir}")
        for lang in tqdm(
            cls.supported_langs, desc=f"Downloading dataset {cls.nickname}"
        ):
            lang_id = cls.lang_to_config.get(lang, lang)
            if not os.path.exists(os.path.join(target_dir, lang_id)):
                logger.info(f"Downloading {lang_id} split...")
                dataset = load_dataset(
                    cls.hf_name, lang_id, num_proc=num_proc, trust_remote_code=True
                )
                dataset.save_to_disk(
                    os.path.join(target_dir, lang_id),
                    max_shard_size="4GB",
                    num_proc=num_proc,
                )
            else:
                logger.info(f"{lang_id} split already exists.")


class GigaSpeechDataset(BaseDataset):
    if_hf_dataset = True
    hf_name = "speechcolab/gigaspeech"
    hf_config = "m"
    is_multilang: bool = False
    lang = "en"
    text_col = "text"
    task = "transcribe"
    remove_columns = [
        "segment_id",
        "speaker",
        "begin_time",
        "end_time",
        "audio_id",
        "title",
        "url",
        "source",
        "category",
    ]


class PeoplesSpeechDataset(BaseDataset):
    is_hf_dataset = True
    nickname = "people_speech"
    hf_name = "MLCommons/peoples_speech/clean"
    hf_config = "clean"
    task = "transcribe"
    lang = "en"
    text_col = "text"
    remove_columns = ["duration_ms", "id"]


class LibriSpeechDataset(BaseDataset):
    is_hf_dataset = True
    nickname = "librispeech"
    hf_name = "openslr/librispeech_asr"
    lang = "en"
    task = "transcribe"
    text_col = "text"
    remove_columns = ["file", "speaker_id", "chapter_id", "id"]
    validation_split_name = ["validation.clean", "validation.other"]


class MultilingualBaseDataset(BaseDataset):

    supported_langs: list[str] = None
    lang_to_config: dict[str, str] = None

    @classmethod
    def get_dataset_dict(cls, splits: list[str], hf_config=None, **kwargs):

        selected_langs = kwargs.get("selected_langs", None)
        if selected_langs is not None:
            target_langs = [l for l in cls.supported_langs if l in selected_langs]
        else:
            target_langs = cls.supported_langs

        assert (
            len(target_langs) > 0
        ), f"No supported languages found in {selected_langs}."

        lang_datasets = list()
        for lang in target_langs:
            if _local_dataset_dir is not None and not kwargs.get(
                "use_load_dataset", False
            ):
                d = load_from_disk(
                    os.path.join(
                        _local_dataset_dir, cls.hf_name, cls.lang_to_config[lang]
                    )
                )
            else:
                d = DatasetDict(
                    {
                        s: load_dataset(
                            cls.hf_name,
                            cls.lang_to_config[lang],
                            split=s,
                            num_proc=kwargs.get("num_proc", 1),
                            trust_remote_code=True,
                            **kwargs.get("load_dataset_kwargs", {}),
                        )
                        for s in splits
                    }
                )

            # Add the length in seconds to each split
            found_splits = d.keys()
            for fs in found_splits:
                d[fs] = add_length_seconds(
                    d[fs],
                    os.path.join(
                        os.getenv("BASEDIR"),
                        f"length_{cls.nickname}_{lang}_{fs}.csv",
                    ),
                )

            # Add the lang and task metadata
            lang_id = (
                cls.mappings2ids.get(lang, lang)
                if hasattr(cls, "mappings2ids")
                else lang
            )

            d = add_lang_task_columns(d, lang_id, cls.task)
            logger.debug(f"Adding to {cls.nickname}: lang {lang_id}, task {cls.task}.")

            # Subsample the validation set if requested
            d = subsample_split(
                d, cls.validation_split_name, kwargs.get("samples_validation", None)
            )
            lang_datasets.append(d)

        data = DatasetDict(
            {s: concatenate_datasets([d[s] for d in lang_datasets]) for s in splits}
        )
        data = data.cast_column("audio", Audio(sampling_rate=16000))
        if cls.text_col != "text":
            data = data.rename_column(cls.text_col, "text")
        data = data.remove_columns(cls.remove_columns)
        return data


# TODO: needs standardization with other classes
class MultilingualLibriSpeechDataset(MultilingualBaseDataset):

    task = "transcribe"
    text_col = "transcript"
    nickname = "mls"
    hf_name = "facebook/multilingual_librispeech"
    validation_split_name = "dev"
    remove_columns = [
        "original_path",
        "begin_time",
        "end_time",
        "file",
        "speaker_id",
        "chapter_id",
        "id",
        "audio_duration",
    ]
    lang_to_config = {
        "nl": "dutch",
        "fr": "french",
        "de": "german",
        "it": "italian",
        "pl": "polish",
        "pt": "portuguese",
        "es": "spanish",
    }
    supported_langs = list(lang_to_config.keys())


class FleursDataset(MultilingualBaseDataset):

    task = "transcribe"
    text_col = "transcription"
    nickname = "fleurs"
    hf_name = "google/fleurs"
    remove_columns = [
        "path",
        "gender",
        "raw_transcription",
        "id",
        "num_samples",
        "lang_id",
        "language",
        "lang_group_id",
    ]
    id2lang = {
        "bg_bg": "bg",  # Bulgarian
        "hr_hr": "hr",  # Croatian
        "cs_cz": "cs",  # Czech
        "da_dk": "da",  # Danish
        "nl_nl": "nl",  # Dutch
        "en_us": "en",  # English
        "et_ee": "et",  # Estonian
        "fi_fi": "fi",  # Finnish
        "fr_fr": "fr",  # French
        "de_de": "de",  # German
        "el_gr": "el",  # Greek
        "hu_hu": "hu",  # Hungarian
        "ga_ie": "ga",  # Irish
        "it_it": "it",  # Italian
        "lv_lv": "lv",  # Latvian
        "lt_lt": "lt",  # Lithuanian
        "mt_mt": "mt",  # Maltese
        "pl_pl": "pl",  # Polish
        "pt_br": "pt",  # Portuguese (assuming pt_br represents EU Portuguese)
        "ro_ro": "ro",  # Romanian
        "sk_sk": "sk",  # Slovak
        "sl_si": "sl",  # Slovene
        "es_419": "es",  # Spanish (assuming es_419 represents EU Spanish)
        "sv_se": "sv",  # Swedish
    }
    supported_langs = list(id2lang.values())
    lang_to_config = {v: k for k, v in id2lang.items()}


class VoxPopuliDataset(MultilingualBaseDataset):

    hf_name = "facebook/voxpopuli"
    task = "transcribe"
    text_col = "raw_text"
    nickname = "voxpopuli"
    remove_columns = [
        "audio_id",
        "language",
        "normalized_text",
        "gender",
        "speaker_id",
        "is_gold_transcript",
        "accent",
    ]

    supported_langs = [
        "en",
        "de",
        "fr",
        "es",
        "pl",
        "it",
        "ro",
        "hu",
        "cs",
        "nl",
        "fi",
        "hr",
        "sk",
        "sl",
        "et",
        "lt",
    ]
    lang_to_config = {k: k for k in supported_langs}


class CommonVoiceDataset(MultilingualBaseDataset):
    hf_name = "mozilla-foundation/common_voice_16_1"
    nickname = "cv16.1"
    task = "transcribe"
    text_col = "sentence"
    remove_columns = [
        "client_id",
        "path",
        "up_votes",
        "down_votes",
        "age",
        "gender",
        "accent",
        "locale",
        "segment",
        "variant",
    ]

    supported_langs = [
        "bg",  # Bulgarian
        "cs",  # Czech
        "da",  # Danish
        "nl",  # Dutch
        "en",  # English
        "et",  # Estonian
        "fi",  # Finnish
        "fr",  # French
        "de",  # German
        "el",  # Greek
        "hu",  # Hungarian
        "ga-IE",  # Irish
        "it",  # Italian
        "lv",  # Latvian
        "lt",  # Lithuanian
        "mt",  # Maltese
        "pl",  # Polish
        "pt",  # Portuguese
        "ro",  # Romanian
        "sk",  # Slovak
        "sl",  # Slovene
        "es",  # Spanish
        "sv-SE",  # Swedish
        "ast",  # Asturian
        "eu",  # Basque
        "br",  # Breton
        "ca",  # Catalan
        "fy-NL",  # Frisian
        "gl",  # Galician
        "oc",  # Occitan
        "rm-sursilv",  # Romansh
        "rm-vallader",  # Romansh
        "sc",  # Sardinian
        "hsb",  # Sorbian
        "cy",  # Welsh
        "zh-CN",  # Chinese (Mandarin)
        "zh-HK",  # Chinese (Cantonese)
        "zh-TW",  # Chinese (Taiwanese Mandarin)
    ]

    mappings2ids = {
        "ga-IE": "ga",
        "sv-SE": "sv",
        "fy-NL": "fy",
        "rm-sursilv": "rm",
        "rm-vallader": "rm",
        "zh-CN": "zh",
        "zh-HK": "zh",
        "zh-TW": "zh",
    }
    lang_to_config = {
        **{k: k for k in supported_langs},
    }


class CoVoST2Dataset(MultilingualBaseDataset):
    nickname = "covost2"
    task = "translate"
    text_col = "translation"

    supported_langs = ["de", "zh"]
    remove_columns = ["sentence", "client_id", "__index_level_0__"]

    @classmethod
    def get_dataset_dict(cls, splits: list[str], hf_config=None, **kwargs):

        selected_langs = kwargs.get("selected_langs", None)
        if selected_langs is not None:
            target_langs = [l for l in cls.supported_langs if l in selected_langs]
        else:
            target_langs = cls.supported_langs

        assert (
            len(target_langs) > 0
        ), f"No supported languages found in {selected_langs}."

        lang_datasets = list()

        # TODO: quick hardcoding, to be removed
        basedir = "/mnt/home/giuseppe/myscratch/speech_lm/datasets/covost2"
        faulty_df = pd.read_csv(os.path.join(basedir, "faulty_files.tsv"), sep="\t")
        faulty_mp3s = set(faulty_df["clip"].tolist())

        for lang in target_langs:
            d = dict()
            for s in splits:
                lang_id = lang if lang != "zh" else "zh-CN"
                s_id = s if s != "validation" else "dev"

                translation_df = pd.read_csv(
                    os.path.join(basedir, f"covost_v2.en_{lang_id}.{s_id}.tsv"),
                    sep="\t",
                    on_bad_lines="warn",
                )
                translation_df = translation_df.rename(columns={"path": "audio"})

                # remove rows with faulty mp3 files (from the CV 4 release)
                # see https://github.com/common-voice/cv-dataset/issues/31
                translation_df = translation_df.loc[
                    ~translation_df["audio"].isin(faulty_mp3s)
                ]

                # create the Audio column linking to local mp3 files
                translation_df["audio"] = translation_df["audio"].apply(
                    lambda x: os.path.join(
                        os.path.dirname(basedir), "cv4", "en", "clips", x
                    )
                )
                ds = Dataset.from_pandas(translation_df).cast_column(
                    "audio", Audio(sampling_rate=16000)
                )
                d[s] = ds

            # Add the length in seconds to each split
            found_splits = d.keys()
            for fs in found_splits:
                d[fs] = add_length_seconds(
                    d[fs],
                    os.path.join(
                        os.getenv("BASEDIR"),
                        f"length_{cls.nickname}_{lang}_{fs}.csv",
                    ),
                )

            # Add the lang and task metadata
            lang_id = (
                cls.mappings2ids.get(lang, lang)
                if hasattr(cls, "mappings2ids")
                else lang
            )

            d = add_lang_task_columns(d, lang_id, cls.task)
            logger.debug(f"Adding to {cls.nickname}: lang {lang_id}, task {cls.task}.")

            # Subsample the validation set if requested
            d = subsample_split(
                d, cls.validation_split_name, kwargs.get("samples_validation", None)
            )
            lang_datasets.append(d)

        data = DatasetDict(
            {s: concatenate_datasets([d[s] for d in lang_datasets]) for s in splits}
        )
        data = data.rename_column(cls.text_col, "text")
        data = data.remove_columns(cls.remove_columns)
        return data


class CommonVoiceSTDataset(MultilingualBaseDataset):
    nickname = "cv16.1-pseudolabel"
    task = "translate"
    text_col = "mt"
    remove_columns = [
        "client_id",
        "sentence",
        "up_votes",
        "down_votes",
        "age",
        "gender",
        "accent",
        "locale",
        "variant",
        "COMET",
        "mt_model",
        "__index_level_0__",
        "segment",
    ]
    supported_langs = ["de", "it", "zh"]

    @classmethod
    def get_dataset_dict(cls, splits: list[str], hf_config=None, **kwargs):

        selected_langs = kwargs.get("selected_langs", None)
        if selected_langs is not None:
            target_langs = [l for l in cls.supported_langs if l in selected_langs]
        else:
            target_langs = cls.supported_langs

        assert (
            len(target_langs) > 0
        ), f"No supported languages found in {selected_langs}."

        lang_datasets = list()

        # TODO: quick hardcoding, to be removed
        basedir = "/mnt/home/giuseppe/myscratch/speech_lm/datasets/covost2"
        faulty_df = pd.read_csv(os.path.join(basedir, "faulty_files.tsv"), sep="\t")
        faulty_mp3s = set(faulty_df["clip"].tolist())

        for lang in target_langs:
            translation_df = pd.read_json(
                f"/mnt/scratch-artemis/bpop/melt-pseudolabeling/commonvoice-en-{lang}-threshold-0.85.jsonl",
                lines=True,
            )
            translation_df = translation_df.rename(columns={"path": "audio"})
            translation_df["audio"] = translation_df["audio"].apply(
                lambda x: os.path.basename(x)
            )

            # remove rows with faulty mp3 files (from the CV 4 release)
            # see https://github.com/common-voice/cv-dataset/issues/31
            translation_df = translation_df.loc[
                ~translation_df["audio"].isin(faulty_mp3s)
            ]

            duration_ms = pd.read_csv(
                os.path.join(
                    os.path.dirname(basedir),
                    "cv-corpus-16.1-2023-12-06",
                    "en",
                    "clip_durations.tsv",
                ),
                sep="\t",
            )
            duration_ms["length"] = duration_ms["duration[ms]"].apply(lambda x: x // 60)

            # Merge the duration data (col: 'clip') with the translation_df (col: 'audio')
            translation_df = translation_df.merge(
                duration_ms,
                left_on="audio",
                right_on="clip",
                how="left",
            )
            # Drop the 'clip' column after merging
            translation_df = translation_df.drop(columns=["clip", "duration[ms]"])

            # create the Audio column linking to local mp3 files
            translation_df["audio"] = translation_df["audio"].apply(
                lambda x: os.path.join(
                    os.path.dirname(basedir),
                    "cv-corpus-16.1-2023-12-06",
                    "en",
                    "clips",
                    x,
                )
            )

            train_df, val_df = train_test_split(
                translation_df, test_size=0.1, random_state=42
            )  # TODO: we could stratify by audio length...

            d = DatasetDict(
                {
                    "train": Dataset.from_pandas(train_df).cast_column(
                        "audio", Audio(sampling_rate=16000)
                    ),
                    "validation": Dataset.from_pandas(val_df).cast_column(
                        "audio", Audio(sampling_rate=16000)
                    ),
                }
            )

            # Add the length in seconds to each split
            found_splits = d.keys()
            for fs in found_splits:
                d[fs] = add_length_seconds(
                    d[fs],
                    os.path.join(
                        os.getenv("BASEDIR"),
                        f"length_{cls.nickname}_{lang}_{fs}.csv",
                    ),
                )

            # Add the lang and task metadata
            lang_id = (
                cls.mappings2ids.get(lang, lang)
                if hasattr(cls, "mappings2ids")
                else lang
            )

            d = add_lang_task_columns(d, lang_id, cls.task)
            logger.debug(f"Adding to {cls.nickname}: lang {lang_id}, task {cls.task}.")

            # Subsample the validation set if requested
            d = subsample_split(
                d, cls.validation_split_name, kwargs.get("samples_validation", None)
            )
            lang_datasets.append(d)

        data = DatasetDict(
            {s: concatenate_datasets([d[s] for d in lang_datasets]) for s in splits}
        )
        data = data.rename_column(cls.text_col, "text")
        data = data.remove_columns(cls.remove_columns)
        return data


#######################
# Spoken QA
#######################


class SpokenSQuADDataset(MultilingualBaseDataset):
    nickname = "spoken-squad"
    task = "sqa"
    text_col = "answer"
    remove_columns = ["id", "context", "__index_level_0__"]
    supported_langs = ["de", "en", "it", "zh"]

    local_dir_template = lambda x: (
        f"/mnt/scratch-artemis/sonal/speech_lm/Spoken-SQuAD/Spoken-SQuAD-data/sqa_{x}.jsonl"
    )

    @classmethod
    def get_dataset_dict(cls, splits: list[str], hf_config=None, **kwargs):
        selected_langs = kwargs.get("selected_langs", None)
        if selected_langs is not None:
            target_langs = [l for l in cls.supported_langs if l in selected_langs]
        else:
            target_langs = cls.supported_langs

        assert (
            len(target_langs) > 0
        ), f"No supported languages found in {selected_langs}."

        lang_datasets = list()

        for lang in target_langs:
            d = dict()
            translation_df = pd.read_json(
                cls.local_dir_template(lang),
                lines=True,
            )
            translation_df = translation_df.rename(columns={"audio_path": "audio"})
            train_df, val_df = train_test_split(
                translation_df, test_size=0.1, random_state=42
            )  # TODO: we could stratify by audio length...

            d = DatasetDict(
                {
                    "train": Dataset.from_pandas(train_df).cast_column(
                        "audio", Audio(sampling_rate=16000)
                    ),
                    "validation": Dataset.from_pandas(val_df).cast_column(
                        "audio", Audio(sampling_rate=16000)
                    ),
                }
            )

            # Add the length in seconds to each split
            found_splits = d.keys()
            for fs in found_splits:
                d[fs] = add_length_seconds(
                    d[fs],
                    os.path.join(
                        os.getenv("BASEDIR"),
                        f"length_{cls.nickname}_{lang}_{fs}.csv",
                    ),
                )

            # Add the lang and task metadata
            lang_id = (
                cls.mappings2ids.get(lang, lang)
                if hasattr(cls, "mappings2ids")
                else lang
            )

            d = add_lang_task_columns(d, lang_id, cls.task)
            logger.debug(f"Adding to {cls.nickname}: lang {lang_id}, task {cls.task}.")

            # Subsample the validation set if requested
            d = subsample_split(
                d, cls.validation_split_name, kwargs.get("samples_validation", None)
            )
            lang_datasets.append(d)

        data = DatasetDict(
            {s: concatenate_datasets([d[s] for d in lang_datasets]) for s in splits}
        )
        data = data.rename_column(cls.text_col, "text")
        data = data.remove_columns(cls.remove_columns)
        return data


class UnanswerableSpokenSQuADDataset(SpokenSQuADDataset):
    nickname = "unanswerable-spoken-squad"
    task = "sqa"
    text_col = "answer"
    remove_columns = ["id", "context", "__index_level_0__"]
    supported_langs = ["de", "en", "it", "zh"]

    local_dir_template = lambda x: (
        f"/mnt/scratch-artemis/sonal/speech_lm/Spoken-SQuAD/Spoken-SQuAD-data/sqa_unanswerable_{x}.jsonl"
    )
