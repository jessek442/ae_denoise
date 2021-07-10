from flask import Flask, render_template, request, send_file
from preprocess import Loader, Padder, LogSpectrogramExtractor, MinMaxNormaliser, Saver, PreprocessingPipeline
from soundgenerator import SoundGenerator
from ae import VAE
import pickle
import os
import soundfile as sf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import tensorflow as tf

app = Flask(__name__)


FRAME_SIZE = 512
HOP_LENGTH = 256
DURATION = 2.97  # in seconds
SAMPLE_RATE = 22050
MONO = True

MODEL_PATH = "./model"

SPECTROGRAM_SAVE_DIR = "./spectrogram"
MIN_MAX_VALUES_SAVE_DIR = "./spectrogram_minmaxvalues"
MINMAXPATH = "./spectrogram_minmaxvalues/min_max_values.pkl"

SAVE_DIR_GENERATED = "./generatedSample/"


def load_fsdd(spectrograms_path):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)  # (n_bins, n_frames, 1)
            x_train.append(spectrogram)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]  # -> (3000, 256, 64, 1)
    return x_train, file_paths


def select_spectrograms(spectrogram,
                        file_path,
                        min_max_values,
                        num_spectrograms):
    sampled_spectrogram = spectrogram
    file_path = file_path
    sampled_min_max_values = [min_max_values[file_path] for file_path in
                              file_path]
    return sampled_spectrogram, sampled_min_max_values


def save_signals(signal, save_dir, sample_rate=22050):
        save_path = os.path.join(save_dir, "clean" + ".wav")
        sf.write(save_path, signal, sample_rate)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/', methods=['POST', 'GET'])
def upload_wav():
    if request.method == "POST":
        print("FORM DATA RECEIVED")

    wavfile = request.files['wavfile']

    wav_path = "./wavfiles/" + wavfile.filename    # save it to the working directory

    wavfile.save(wav_path)

    # instantiate objects for loading and extracting features from uploaded audio file

    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    normalizer = MinMaxNormaliser(0, 1)
    saver = Saver(SPECTROGRAM_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

    preprocessor = PreprocessingPipeline()

    preprocessor.loader = loader
    preprocessor.padder = padder
    preprocessor.extractor = extractor
    preprocessor.normaliser = normalizer
    preprocessor.saver = saver

    feature_min_max_values = preprocessor._process_file(wav_path)     # Process the wavfile uploaded by the user and have it saved into ./spectrogram
    preprocessor.saver.save_min_max_values(preprocessor.min_max_values)

    # GENERATE THE SAMPLE TO BE RETURNED TO THE USER FOR DOWNLOAD

    # initialise sound generator
    vae = VAE.load(MODEL_PATH)
    sound_generator = SoundGenerator(vae, HOP_LENGTH)

    # load spectrograms + min max values
    with open(MINMAXPATH, "rb") as f:
        min_max_values = pickle.load(f)


    spec = np.load("./spectrogram/" + wavfile.filename + ".npy")
    spectro_path = "./spectrogram/" + wavfile.filename + ".npy"
    print(spectro_path)

    spec = spec[..., np.newaxis]
    spec = spec[np.newaxis, :, :, :]
    print(spec.shape)
    spec_min_max = feature_min_max_values

    print(spec_min_max[0], spec_min_max[1])

    """ Select Spectrograms is returning the spectrogram and the min max values
    sampled_spec, sampled_min_max_values = select_spectrograms(spec,
                                                               spectro_path,
                                                               min_max_values,
                                                               1)
    """

    signal, _ = sound_generator.generate(spec,
                                         spec_min_max)

    save_signals(signal, SAVE_DIR_GENERATED)



    return render_template('index.html')

@app.route('/download')
def download_file():
    path = "./generatedSample/clean.wav"
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run()
