import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from luigi_openslr.luigi_config import PathConfig


def flac2ndarray(filepath):
    return sf.read(filepath)


def dict_audio_to_matrix(dict_audio):
    logger = logging.getLogger('training')
    list_index = []
    list_samples = []
    min_len = float('inf')
    for key_speaker, list_audio in dict_audio.items():
        for value_audio in list_audio:
            if len(value_audio) > 1000:
                if len(value_audio) < min_len:
                    min_len = len(value_audio)
                list_index.append(key_speaker)
                list_samples.append(value_audio)
    logger.debug('Minimal length: {}'.format(min_len))
    logger.debug('Number of samples kept: {}'.format(len(list_samples)))

    matrix_audio = np.asarray([sample[:min_len] for sample in list_samples])
    logger.debug('Matrix audio: {}'.format(matrix_audio.shape))
    return list_index, matrix_audio


def id_to_variable(df, variable_name):
    unique_variable = pd.unique(df[variable_name])
    variable_to_id = {value: [] for value in unique_variable}
    id2variable = {id_speaker: [] for id_speaker in df['ID']}
    for row in df[['ID', variable_name]].iterrows():
        _, ser = row
        id_speaker, value = ser['ID'], ser[variable_name]
        variable_to_id[value].append(id_speaker)
        id2variable[id_speaker].append(value)
    return id2variable


def loading_audio_files(train_id):
    logger = logging.getLogger('training')
    X_dict = {}
    unique_sampling_rate = set()
    # Look at every speaker folder
    for speaker_folder in glob.iglob(PathConfig().librispeech_path + '/*'):
        path_speak = Path(speaker_folder).stem
        # If we want to load this speaker files
        if path_speak in train_id:
            X_dict[int(path_speak)] = []
            # For every chapter
            for chapter_path in glob.iglob(speaker_folder + '/*'):
                # Load every audio file
                for file_audio in glob.iglob(chapter_path + '/*.flac'):
                    vec_audio, sampling_rate = flac2ndarray(file_audio)
                    unique_sampling_rate.add(sampling_rate)
                    X_dict[int(path_speak)].append(vec_audio)
    return X_dict