import os.path

from pydub import AudioSegment
import numpy as np
import librosa
import soundfile as sf
from pyo import FFT, SfPlayer, Server, ExpTable, TableIndex, IFFT, savefile, sndinfo, CarToPol, PolToCar


def main():
    title = "elive151221"
    audio_seg = AudioSegment.from_wav(f"{title}.wav")
    export_to_mp3(audio_seg, f"{title}", 320)
    subtract_spectral_stereo_pyo_2(f"{title}.wav", f"{title}.mp3", "elive_fucking.wav")



def export_to_mp3(audio_segment, title, bitrate):
    audio_segment.export(f"{title}.mp3", format="mp3", bitrate=f"{bitrate}k")
    audio_seg = AudioSegment.from_mp3(f"{title}.mp3")
    audio_seg.export(f"{title}mp3.wav", format="wav")


def subtract_spectral_stereo_pyo_1(source, to_subtract, output_filename):
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
    mag = pol1["mag"] * pol2["mag"]
    pha = pol1["ang"] + pol2["ang"] * -1
    # converts back to rectangular
    car = PolToCar(mag, pha)

    re_final = re_source - re_to_subtract
    im_final = im_source - im_to_subtract
    fout_final = IFFT(car["real"], car["imag"], size=1024, overlaps=4, wintype=2)

    fout_final.out()
    s.start()

def subtract_spectral_stereo_pyo_2(source, to_subtract, output_filename):
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
    mag = pol1["mag"] * pol2["mag"]
    pha = pol1["ang"] + pol2["ang"]
    # converts back to rectangular
    car = PolToCar(mag, pha)

    re_final = re_source - re_to_subtract
    im_final = im_source - im_to_subtract
    fout_final = IFFT(car["real"], car["imag"], size=1024, overlaps=4, wintype=2)

    fout_final.out()
    s.start()

def subtract_spectral_stereo_pyo_3(source, to_subtract, output_filename, winsize, wintype):
    s = Server(duplex=0, audio="offline").boot()

    filedur = sndinfo(source)[1]
    s.recordOptions(dur=filedur, filename=output_filename)

    sf_source = SfPlayer(source)
    fin_source = FFT(sf_source, size=winsize, overlaps=4, wintype=wintype)
    t_source = ExpTable([(0,0),(int(winsize/2),0)], size=int(winsize/2))
    amp_source = TableIndex(t_source, fin_source["bin"])
    re_source = fin_source["real"] * amp_source
    im_source = fin_source["imag"] * amp_source

    to_subtract_filename_split = os.path.splitext(to_subtract)
    to_subtract_new_filename = f"{to_subtract_filename_split[0]}mp3.wav"
    sf_to_subtract = SfPlayer(to_subtract_new_filename)
    fin_to_subtract = FFT(sf_to_subtract, size=winsize, overlaps=4, wintype=wintype)
    t_to_subtract = ExpTable([(0,0),(int(winsize/2),0)], size=int(winsize/2))
    amp_to_subtract = TableIndex(t_to_subtract, fin_to_subtract["bin"])
    re_to_subtract = fin_to_subtract["real"] * amp_to_subtract
    im_to_subtract = fin_to_subtract["imag"] * amp_to_subtract

    pol1 = CarToPol(fin_source["real"], fin_source["imag"])
    pol2 = CarToPol(fin_to_subtract["real"], fin_to_subtract["imag"])
    # times magnitudes and adds phases
    mag = (pol1["mag"] - pol2["mag"])
    pha = pol1["ang"]
    # converts back to rectangular
    car = PolToCar(mag, pha)

    re_final = re_source - re_to_subtract
    im_final = im_source - im_to_subtract
    fout_final = IFFT(car["real"], car["imag"], size=winsize, overlaps=4, wintype=wintype)

    fout_final.out()
    s.start()


if __name__ == '__main__':
    main()
