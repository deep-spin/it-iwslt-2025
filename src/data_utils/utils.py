from datasets import concatenate_datasets, DatasetDict, IterableDataset, Dataset
from .supported_datasets import (
    LibriSpeechDataset,
    PeoplesSpeechDataset,
    MultilingualLibriSpeechDataset,
    FleursDataset,
    VoxPopuliDataset,
    CommonVoiceDataset,
    CoVoST2Dataset,
    CommonVoiceSTDataset,
    SpokenSQuADDataset,
    UnanswerableSpokenSQuADDataset,
)
import os
from . import _local_dataset_dir


def get_dataset(dataset_name: str, splits: list[str], n_proc: int = 4, **kwargs):
    """Training data loading utility.

    From a simple string identifier, it loads a dataset from the Hugging Face Hub using the
    `datasets` library, and returns a `DatasetDict` object.
    Crucially, the function decides how the training and validation split in the code base is constructed.
    """

    kwargs = {"num_proc": n_proc, "splits": splits, "trust_remote_code": True, **kwargs}
    if dataset_name == LibriSpeechDataset.nickname:

        new_splits = splits.copy()
        if "train" in new_splits:
            new_splits.remove("train")
            new_splits.extend(["train.clean.100", "train.clean.360", "train.other.500"])
        if "validation" in new_splits:
            new_splits.remove("validation")
            new_splits.extend(["validation.clean", "validation.other"])
        if "test" in new_splits:
            new_splits.remove("test")
            new_splits.extend(["test.clean", "test.other"])

        kwargs["splits"] = new_splits

        data = LibriSpeechDataset.get_dataset_dict(**kwargs)

        if "train" in splits:
            data["train"] = concatenate_datasets(
                [
                    data.pop("train.clean.100"),
                    data.pop("train.clean.360"),
                    data.pop("train.other.500"),
                ]
            )
        if "validation" in splits:
            data["validation"] = concatenate_datasets(
                [data.pop("validation.clean"), data.pop("validation.other")]
            )

    elif dataset_name == PeoplesSpeechDataset.nickname:
        data = PeoplesSpeechDataset.get_dataset_dict(**kwargs)
    # elif dataset_name == "gigaspeech":
    # data = GigaSpeechDataset.get_dataset_dict(**kwargs)
    elif dataset_name == MultilingualLibriSpeechDataset.nickname:
        if "validation" in splits:
            splits.remove("validation")
            splits.append("dev")
            kwargs["splits"] = splits
        data = MultilingualLibriSpeechDataset.get_dataset_dict(**kwargs)
        if "dev" in data:
            data["validation"] = data.pop("dev")
    elif dataset_name == FleursDataset.nickname:
        data = FleursDataset.get_dataset_dict(**kwargs)
    elif dataset_name == VoxPopuliDataset.nickname:
        data = VoxPopuliDataset.get_dataset_dict(**kwargs)
    elif dataset_name == CommonVoiceDataset.nickname:
        data = CommonVoiceDataset.get_dataset_dict(**kwargs)
    elif dataset_name == CoVoST2Dataset.nickname:
        data = CoVoST2Dataset.get_dataset_dict(**kwargs)
    elif dataset_name == CommonVoiceSTDataset.nickname:
        data = CommonVoiceSTDataset.get_dataset_dict(**kwargs)
    elif dataset_name == SpokenSQuADDataset.nickname:
        data = SpokenSQuADDataset.get_dataset_dict(**kwargs)
    elif dataset_name == UnanswerableSpokenSQuADDataset.nickname:
        data = UnanswerableSpokenSQuADDataset.get_dataset_dict(**kwargs)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    return data
