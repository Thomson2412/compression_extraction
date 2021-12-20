import os.path

from pydub import AudioSegment
import numpy as np
import librosa
import soundfile as sf
from pyo import FFT, SfPlayer, Server, ExpTable, TableIndex, IFFT, savefile, sndinfo, CarToPol, PolToCar


def main():
    title = "sine1111"
    audio_seg = AudioSegment.from_wav(f"{title}.wav")
    export_to_mp3(audio_seg, f"{title}", 8)
    subtract_spectral_stereo_pyo(f"{title}.wav", f"{title}.mp3", "dsjakndakjdhksad2old.py.wav")
    #subtract_spectral_mono("sine1111.wav", "mp3.mp3", "sine2subtracted_spectral_mono.wav")
    #subtract_spectral_stereo("sine1111.wav", "mp3.mp3", "sine2subtracted_spectral_stereo.wav")
    # subtract_raw_pydub("sine1111.wav", "sine1111.mp3", "subtracted_pydub.wav")
    # subtract_raw_librosa("sine1111.wav", "sine1111.mp3", "subtracted_librosa.wav")


def export_to_mp3(audio_segment, title, bitrate):
    audio_segment.export(f"{title}.mp3", format="mp3", bitrate=f"{bitrate}k")
    audio_seg = AudioSegment.from_mp3(f"{title}.mp3")
    audio_seg.export(f"{title}mp3.wav", format="wav")


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


def subtract_spectral_stereo_pyo(source, to_subtract, output_filename):
    s = Server(duplex=0, audio="offline").boot()

    filedur = sndinfo(source)[1]
    s.recordOptions(dur=filedur, filename=output_filename)

    sf_source = SfPlayer(source)
    fin_source = FFT(sf_source, size=1024, overlaps=4, wintype=2)
    t_source = ExpTable([(0,0),(3,0),(10,1),(20,0),(30,.8),(50,0),(70,.6),(150,0),(512,0)], size=512)
    amp_source = TableIndex(t_source, fin_source["bin"])
    re_source = fin_source["real"] * amp_source
    im_source = fin_source["imag"] * amp_source

    to_subtract_filename_split = os.path.splitext(to_subtract)
    to_subtract_new_filename = f"{to_subtract_filename_split[0]}mp3.wav"
    sf_to_subtract = SfPlayer(to_subtract_new_filename)
    fin_to_subtract = FFT(sf_to_subtract, size=1024, overlaps=4, wintype=2)
    t_to_subtract = ExpTable([(0,0),(3,0),(10,1),(20,0),(30,.8),(50,0),(70,.6),(150,0),(512,0)], size=512)
    amp_to_subtract = TableIndex(t_to_subtract, fin_to_subtract["bin"])
    re_to_subtract = fin_to_subtract["real"] * amp_to_subtract
    im_to_subtract = fin_to_subtract["imag"] * amp_to_subtract

    pol1 = CarToPol(fin_source["real"], fin_source["imag"])
    pol2 = CarToPol(fin_to_subtract["real"], fin_to_subtract["imag"])
    # times magnitudes and adds phases
    mag = pol1["mag"] * pol2["mag"] * 100
    pha = pol1["ang"] + pol2["ang"] * -1
    # converts back to rectangular
    car = PolToCar(mag, pha)

    re_final = re_source - re_to_subtract
    im_final = im_source - im_to_subtract
    fout_final = IFFT(car["real"], car["imag"], size=1024, overlaps=4, wintype=2)

    fout_final.out()
    s.start()


if __name__ == '__main__':
    main()
