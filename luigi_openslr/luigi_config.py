from pathlib import Path

import luigi

URL_OPENSLR = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
CURRENT_FOLDER = Path(__file__).parent
DATA_FOLDER = CURRENT_FOLDER.parent.joinpath('data')
MODEL_FOLDER = DATA_FOLDER.joinpath('models')


class PathConfig(luigi.Config):
    dev_clean = luigi.Parameter(default=str(DATA_FOLDER.joinpath('dev-clean.tar.gz')))
    librispeech_path = luigi.Parameter(default=str(DATA_FOLDER.joinpath('LibriSpeech/dev-clean/')))
    speakers_data = luigi.Parameter(default=str(DATA_FOLDER.joinpath('LibriSpeech/SPEAKERS.TXT')))

    speaker_pck = luigi.Parameter(default=str(DATA_FOLDER.joinpath('speakers.pck')))
    id_to_variable = luigi.Parameter(default=str(DATA_FOLDER.joinpath('id_to_variable.json')))
    train_id2variable = luigi.Parameter(default=str(DATA_FOLDER.joinpath('id_to_variable.json')))
    test_id2variable = luigi.Parameter(default=str(DATA_FOLDER.joinpath('id_to_variable.json')))

    DATA_FOLDER.mkdir('matrices/')
    matrix_folder = DATA_FOLDER.joinpath('matrices')

    x_train_mat = luigi.Parameter(default=str(matrix_folder.joinpath('xtrain.npz')))
    y_train_mat = luigi.Parameter(default=str(matrix_folder.joinpath('ytrain.npz')))
    x_test_mat = luigi.Parameter(default=str(matrix_folder.joinpath('xtest.npz')))
    y_test_mat = luigi.Parameter(default=str(matrix_folder.joinpath('ytest.npz')))

    model_def_path = luigi.Parameter(default=str(MODEL_FOLDER.joinpath('definition.json')))
    model_weights_path = luigi.Parameter(default=str(MODEL_FOLDER.joinpath('trained_weights.h5')))