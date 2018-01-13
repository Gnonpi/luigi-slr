import shutil
import tarfile
from pathlib import Path

import logging
import luigi
import pandas as pd
import numpy as np
import requests
from luigi import Task
from luigi.local_target import LocalTarget

from luigi_openslr.luigi_config import PathConfig, URL_OPENSLR, DATA_FOLDER
from luigi_openslr.utils.audio import dict_audio_to_matrix, id_to_variable, loading_audio_files


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
        logger = logging.getLogger('training')
        logger.info('Downloading archive')
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
        logger = logging.getLogger('training')
        path_librispeech = Path(PathConfig().librispeech_path)
        if not path_librispeech.exists():
            logger.info('Extracting archive')
            tar = tarfile.open(PathConfig().dev_clean)
            tar.extractall(path=str(DATA_FOLDER))
            tar.close()


class LoadSpeakerInfo(Task):
    def requires(self):
        return ExtractTargz()

    def output(self):
        return luigi.LocalTarget(PathConfig().speaker_pck)

    def run(self):
        logger = logging.getLogger('training')
        logger.info('Loading speaker file')
        bad_speaker_name = ('|CBW|Simon', 'CBW-Simon')
        with self.input().open('r') as f_speaker:
            with open('tmp', 'w') as tmp:
                for line in f_speaker:
                    if bad_speaker_name[0] in line:
                        line = line.replace(bad_speaker_name[0], bad_speaker_name[1])
                    tmp.write(line)
        logger.info('Dumping to csv')
        df = pd.read_csv('tmp',
                         sep='|',
                         comment=';')
        df.columns = ['ID', 'SEX', 'SUBSET', 'MINUTES', 'NAME']
        df.to_pickle(self.output().path)


class SplitTrainAndTest(luigi.Task):
    variable_prediction = luigi.Parameter()
    ratio_split = luigi.FloatParameter(default=0.7)

    def requires(self):
        return LoadSpeakerInfo()

    def output(self):
        return {
            'train_var': luigi.LocalTarget(PathConfig().train_id2variable),
            'test_var': luigi.LocalTarget(PathConfig().test_id2variable)
        }

    def run(self):
        logger = logging.getLogger('training')
        logger.info('Splitting in train/test')
        df_speaker = pd.read_pickle(self.input().path)
        df_id2variable = df_speaker[['ID', str(self.variable_prediction)]]
        msk = np.random.rand(len(df_id2variable)) < self.ratio_split
        df_train = df_id2variable[msk]
        df_test = df_id2variable[~msk]

        logger.debug('Saving train set')
        with open(self.output()['train_var'].path, 'wb') as f:
            df_train.to_pickle(f)

        logger.debug('Saving test set')
        with open(self.output()['test_var'].path, 'wb') as f:
            df_test.to_pickle(f)


class AudioToMatrix(luigi.Task):
    variable_prediction = luigi.Parameter()

    def output(self):
        return {
            'X_train': luigi.LocalTarget(PathConfig().x_train_mat),
            'y_train': luigi.LocalTarget(PathConfig().y_train_mat),
            'X_test': luigi.LocalTarget(PathConfig().x_test_mat),
            'y_test': luigi.LocalTarget(PathConfig().y_test_mat),
        }

    def requires(self):
        return SplitTrainAndTest(self.variable_prediction)

    def run(self):
        df_train_id2variable = pd.read_pickle(self.input()['train_var'].path)
        # y_train = df_train_id2variable[str(self.variable_prediction)]

        train_id = df_train_id2variable['ID'].tolist()
        train_id = [str(id_) for id_ in train_id]

        X_dict = loading_audio_files(train_id)

        list_index, X_train = dict_audio_to_matrix(X_dict)
        id2variable = id_to_variable(df_train_id2variable, str(self.variable_prediction))
        y_train = np.asarray([id2variable[id_speaker] for id_speaker in list_index])