from pathlib import Path

import luigi

URL_OPENSLR = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
CURRENT_FOLDER = Path(__file__).parent
DATA_FOLDER = CURRENT_FOLDER.parent.joinpath('data')


class PathConfig(luigi.Config):
    dev_clean = luigi.Parameter(default=str(DATA_FOLDER.joinpath('dev-clean.tar.gz')))
    librispeech_path = luigi.Parameter(default=str(DATA_FOLDER.joinpath('LibriSpeech/dev-clean/')))
    speakers_data = luigi.Parameter(default=str(DATA_FOLDER.joinpath('LibriSpeech/SPEAKERS.TXT')))

    speaker_npy = luigi.Parameter(default=str(DATA_FOLDER.joinpath('speakers.npy')))
    dev_npy = luigi.Parameter(default=str(DATA_FOLDER.joinpath('luigi-load.npy')))
