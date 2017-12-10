import os

import luigi

from download_set import download_dev_set, untar_dev_set
from luigi_config import PathConfig


class LoadDataset(luigi.Task):
    def output(self):
        return luigi.LocalTarget(PathConfig().dev_npy)

    def requires(self):
        return

    def run(self):
        if not PathConfig().dev_npy.exists():
            if not PathConfig().dev_clean.exists():
                download_dev_set()
            untar_dev_set()


class DoStuff(luigi.Task):
    def output(self):
        return luigi.LocalTarget(PathConfig().data_folder.joinpath('blabla'))

    def requires(self):
        return LoadDataset()

    def run(self):
        for i in range(10):
            print(i)
