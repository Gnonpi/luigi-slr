import os
import tarfile

import traceback
import urllib

import luigi
import requests
import shutil
from luigi import Task
from luigi.local_target import LocalFileSystem, LocalTarget
from pathlib import Path

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
        # NOTE the stream=True parameter
        with requests.get(URL_OPENSLR, stream=True) as response:
            with open('tmp', 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        # f.flush() commented by recommendation from J.F.Sebastian
        shutil.move('tmp', self.output().path)


class ExtractTargz(luigi.Task):
    """
    Extract the files from the tar gz they are in
    """

    def requires(self):
        return DownloadDatasetTargz()

    def output(self):
        path_one_file = Path(PathConfig().librispeech_path).joinpath('dev-clean',
                                                                     '8842',
                                                                     '304647',
                                                                     '8842-304647-0013.flac')
        return LocalTarget(str(path_one_file))

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
        return luigi.LocalTarget(PathConfig().dev_npy)

    def run(self):

