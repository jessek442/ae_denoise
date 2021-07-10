import librosa
from preprocess import MinMaxNormaliser



class SoundGenerator:
    """SoundGenerator is responsible for generating audios from
    spectrograms.
    """

    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length
        self._min_max_normaliser = MinMaxNormaliser(0, 1)

    def generate(self, spectrogram, min_max_values):
        generated_spectrogram, latent_representations = \
            self.vae.reconstruct(spectrogram)

        print("gen_spec_shape", generated_spectrogram.shape)
        generated_spectrogram=generated_spectrogram[0, :, :, :]
        print("gen_spec_shape", generated_spectrogram.shape)
        signal = self.convert_spectrogram_to_audio(generated_spectrogram, min_max_values)
        return signal, latent_representations

    def convert_spectrogram_to_audio(self, spectrogram, min_max_values):
        # reshape the log spectrogram
        log_spectrogram = spectrogram[:, :, 0]
        # apply denormalisation
        denorm_log_spec = self._min_max_normaliser.denormalise(
            log_spectrogram, min_max_values[0], min_max_values[1])
        # log spectrogram -> spectrogram
        spec = librosa.db_to_amplitude(denorm_log_spec)
        # apply Griffin-Lim
        signal = librosa.istft(spec, hop_length=self.hop_length)
        return signal