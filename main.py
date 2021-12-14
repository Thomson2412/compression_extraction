from pydub import AudioSegment
import numpy as np
import librosa
import soundfile as sf


def main():
    audio_seg = AudioSegment.from_wav("horord.wav")
    export_to_mp3(audio_seg, "horord", 256)

    subtract_spectral_mono("horord.wav", "horord.mp3", "subtracted_spectral_mono.wav")
    subtract_spectral_stereo("horord.wav", "horord.mp3", "subtracted_spectral_stereo.wav")
    subtract_raw_pydub("horord.wav", "horord.mp3", "subtracted_pydub.wav")
    subtract_raw_librosa("horord.wav", "horord.mp3", "subtracted_librosa.wav")


def export_to_mp3(audio_segment, title, bitrate):
    audio_segment.export(f"{title}.mp3", format="mp3", bitrate=f"{bitrate}k")


def subtract_raw_pydub(source, to_subtract, output_filename):
    source_seg = AudioSegment.from_wav(source)
    to_subtract_seg = AudioSegment.from_mp3(to_subtract)

    source_seg = source_seg.set_sample_width(4)
    to_subtract_seg = to_subtract_seg.set_sample_width(4)

    if source_seg.duration_seconds != to_subtract_seg.duration_seconds:
        raise Exception("Need the same duration")
    if source_seg.frame_rate != to_subtract_seg.frame_rate:
        raise Exception("Need the same sample rate")
    if source_seg.channels != to_subtract_seg.channels:
        raise Exception("Need the same number of channels")

    source_arr = np.array(source_seg.get_array_of_samples())
    if source_seg.channels == 2:
        source_arr = source_arr.reshape((-1, 2))
    source_arr = np.float32(source_arr) / 2 ** 15

    to_subtract_arr = np.array(to_subtract_seg.get_array_of_samples())
    if source_seg.channels == 2:
        to_subtract_arr = to_subtract_arr.reshape((-1, 2))
    to_subtract_arr = np.float32(to_subtract_arr) / 2 ** 15

    if len(source_arr) != len(to_subtract_arr):
        raise Exception("Need the same length of arrays")

    subtracted_arr = source_arr - to_subtract_arr

    y = np.int16(subtracted_arr)
    output_song = AudioSegment(
        y.tobytes(),
        frame_rate=source_seg.frame_rate,
        sample_width=2,
        channels=source_seg.channels)
    output_song.export(output_filename, format="wav")


def subtract_raw_librosa(source, to_subtract, output_filename):
    source_arr, s_sr = librosa.load(source, sr=None, mono=False)
    to_subtract_arr, ts_sr = librosa.load(to_subtract, sr=None, mono=False)

    if len(source_arr) != len(to_subtract_arr):
        raise Exception("Need the same length of arrays")
    if s_sr != ts_sr:
        raise Exception("Need the same sample rate")

    subtracted_arr = source_arr - to_subtract_arr

    sf.write(output_filename, subtracted_arr.T, s_sr, "PCM_24")


def subtract_spectral_stereo(source, to_subtract, output_filename):
    # load input file, and stft (Short-time Fourier transform)
    source_arr, s_sr = librosa.load(source, sr=None, mono=False)  # keep native sr (sampling rate) and trans into mono

    # Left channel
    source_stft_l = librosa.stft(source_arr[0])  # Short-time Fourier transform
    source_magnitude_l = np.abs(source_stft_l)  # get magnitude
    source_phase_l = np.angle(source_stft_l)  # get phase
    source_phase_inverse_l = np.exp(1.0j * source_phase_l)  # use this phase information when Inverse Transform

    # Right channel
    source_stft_r = librosa.stft(source_arr[1])  # Short-time Fourier transform
    source_magnitude_r = np.abs(source_stft_r)  # get magnitude
    source_phase_r = np.angle(source_stft_r)  # get phase
    source_phase_inverse_r = np.exp(1.0j * source_phase_r)  # use this phase information when Inverse Transform

    # load noise only file, stft
    to_subtract_arr, ts_sr = librosa.load(to_subtract, sr=None, mono=False)

    # Left channel
    to_subtract_stft_l = librosa.stft(to_subtract_arr[0])
    to_subtract_magnitude_l = np.abs(to_subtract_stft_l)

    # Right channel
    to_subtract_stft_r = librosa.stft(to_subtract_arr[1])
    to_subtract_magnitude_r = np.abs(to_subtract_stft_r)

    # subtract noise spectral mean from input spectral, and istft (Inverse Short-Time Fourier Transform)
    # subtracted_arr = source_stft_abs - to_subtract_stft_abs_mean.reshape((to_subtract_stft_abs_mean.shape[0], 1))  # reshape for broadcast to subtract
    subtracted_mag_l = source_magnitude_l - to_subtract_magnitude_l
    subtracted_mag_r = source_magnitude_r - to_subtract_magnitude_r

    subtracted_arr_inverse_l = subtracted_mag_l * source_phase_inverse_l  # apply phase information
    subtracted_arr_inverse_r = subtracted_mag_r * source_phase_inverse_r  # apply phase information
    y_l = librosa.istft(subtracted_arr_inverse_l)  # back to time domain signal
    y_r = librosa.istft(subtracted_arr_inverse_r)  # back to time domain signal

    # save as a wav file
    sf.write(output_filename, np.array((y_l, y_r)).T, s_sr, "PCM_24")


def subtract_spectral_mono(source, to_subtract, output_filename):
    # load input file, and stft (Short-time Fourier transform)
    source_arr, s_sr = librosa.load(source, sr=None, mono=True)  # keep native sr (sampling rate) and trans into mono

    source_stft = librosa.stft(source_arr)  # Short-time Fourier transform
    source_magnitude = np.abs(source_stft)  # get magnitude
    source_phase = np.angle(source_stft)  # get phase
    source_phase_inverse = np.exp(1.0j * source_phase)  # use this phase information when Inverse Transform

    # load noise only file, stft
    to_subtract_arr, ts_sr = librosa.load(to_subtract, sr=None, mono=True)

    to_subtract_stft = librosa.stft(to_subtract_arr)
    to_subtract_magnitude = np.abs(to_subtract_stft)

    # subtract noise spectral mean from input spectral, and istft (Inverse Short-Time Fourier Transform)
    # subtracted_arr = source_stft_abs - to_subtract_stft_abs_mean.reshape((to_subtract_stft_abs_mean.shape[0], 1))  # reshape for broadcast to subtract
    subtracted_mag = source_magnitude - to_subtract_magnitude

    subtracted_arr_inverse = subtracted_mag * source_phase_inverse  # apply phase information
    y = librosa.istft(subtracted_arr_inverse)  # back to time domain signal

    # save as a wav file
    sf.write(output_filename, y, s_sr, "PCM_24")


if __name__ == '__main__':
    main()
