import os

EU_LANGUAGES_ISO = [
    "bg",  # Bulgarian
    "hr",  # Croatian
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
    "ga",  # Irish
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
    "sv",  # Swedish
]

OTHER_LANGUAGES_ISO = [
    "ca",  # Catalan
    "sq",  # Albanian
]

_local_dataset_dir = os.environ.get("LOCAL_DATASETS_DIR", None)
