import logging

import luigi
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.models import model_from_json

from luigi_openslr.load_tasks import AudioToMatrix
from luigi_openslr.luigi_config import PathConfig


class SetUpLstm(luigi.Task):
    pts_start = luigi.IntParameter(default=2000)

    def output(self):
        return luigi.LocalTarget(PathConfig().model_def_path)

    def run(self):
        logger = logging.getLogger('training')
        logger.info('Setting up neural model')
        model = Sequential()

        model.add(LSTM(64,
                       activation='relu',
                       return_sequences=True,
                       input_shape=(1, 24080)))
        model.add(LSTM(32,
                       activation='relu',
                       return_sequences=True))
        model.add(LSTM(16,
                       activation='relu'))

        model.add(Dense(24, activation='tanh'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(12, activation='tanh'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        model_json = model.to_json()
        logger.debug('Saving model definition to json')
        with self.output().open('w') as json_file:
            json_file.write(model_json)


class TrainModel(luigi.Task):
    variable_prediction = luigi.Parameter(default='SEX')

    def output(self):
        return luigi.LocalTarget(PathConfig().model_weights_path)

    def requires(self):
        return {'setup_lstm': SetUpLstm(),
                'audio2matrix': AudioToMatrix(variable_prediction=self.variable_prediction)}

    def run(self):
        logger = logging.getLogger('training')
        logger.info('Training DL model')
        X_train = self.input()['audio2matrix']['X_train']
        y_train = self.input()['audio2matrix']['y_train']

        nb_samples, length_audio = X_train.shape
        X_train = X_train.reshape(nb_samples, 1, length_audio)

        with self.input()['setup_lstm'].open('r') as json_file:
            model = model_from_json(json_file.read())
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        logger.info('Model input shape: {}'.format(model.input_shape))
        logger.info('Model input shape: {}'.format(model.output_shape))

        model.fit(X_train,
                  y_train,
                  batch_size=64,
                  epochs=10)

