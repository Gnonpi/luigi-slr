import fileinput
import shutil
import tarfile
from pathlib import Path

import luigi
import pandas as pd
import numpy as np
import requests
from luigi import Task
from luigi.local_target import LocalTarget

from luigi_openslr.luigi_config import PathConfig, URL_OPENSLR, DATA_FOLDER


class DownloadDatasetTargz(luigi.Task):
    """
    Download the Open SLR dev-clean dataset
    """
    def requires(self):
        pass

    def output(self):
        return LocalTarget(PathConfig().dev_clean)

    def run(self):
        # luigi.LocalTarget cannot be opened with 'wb' mode
        # so I download in a tmp file and then move it
        with requests.get(URL_OPENSLR, stream=True) as response:
            with open('tmp', 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move('tmp', self.output().path)


class ExtractTargz(luigi.Task):
    """
    Extract the files from the tar gz they are in
    """

    def requires(self):
        return DownloadDatasetTargz()

    def complete(self):
        # A LocalFileSystem cannot be used because of a missing .exists method
        path_one_file = Path(PathConfig().librispeech_path).joinpath('8842',
                                                                     '304647',
                                                                     '8842-304647-0013.flac')
        return path_one_file.exists()

    def output(self):
        return LocalTarget(PathConfig().speakers_data)

    def run(self):
        path_librispeech = Path(PathConfig().librispeech_path)
        if not path_librispeech.exists():
            tar = tarfile.open(PathConfig().dev_clean)
            tar.extractall(path=str(DATA_FOLDER))
            tar.close()


class LoadDataset(Task):
    def requires(self):
        return ExtractTargz()

    def output(self):
        return luigi.LocalTarget(PathConfig().speaker_npy)

    def run(self):
        bad_speaker_name = ('|CBW|Simon', 'CBW-Simon')
        with self.input().open('r') as f_speaker:
            with open('tmp', 'w') as tmp:
                for line in f_speaker:
                    if bad_speaker_name[0] in line:
                        line = line.replace(bad_speaker_name[0], bad_speaker_name[1])
                    tmp.write(line)
        df = pd.read_csv('tmp',
                         sep='|',
                         comment=';')
        df.columns = ['ID', 'SEX', 'SUBSET', 'MINUTES', 'NAME']
        np.save(self.output().path, df.as_matrix())
