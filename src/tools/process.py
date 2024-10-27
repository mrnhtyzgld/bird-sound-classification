from os import path 
import os
from pydub import AudioSegment 

from scipy import signal
from scipy.io import wavfile
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt

def convert_mp3_to_wav(input_directory, output_directory):
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Iterate over all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".mp3"):
            mp3_path = os.path.join(input_directory, filename)
            wav_filename = os.path.splitext(filename)[0] + ".wav"
            wav_path = os.path.join(output_directory, wav_filename)
            
            # Load the MP3 file and convert it to WAV
            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(wav_path, format="wav")


def spec(input_directory,output_directory,min_db = -20,max_db = 150,min_fq = 0,max_fq = 22000):
    # Read the WAV file
    for filename in os.listdir(input_directory):
        if filename.endswith(".wav"):
            wav_path = os.path.join(input_directory, filename)
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_path = os.path.join(output_directory, png_filename)

            sample_rate, samples = wavfile.read(wav_path)

            # If the audio has more than one channel, convert it to mono by averaging
            if len(samples.shape) > 1:
                samples = samples.mean(axis=1)

            frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=256)

            # Convert the spectrogram to dB scale
            spectrogram_db = 10 * np.log10(spectrogram)

            # Filter out frequencies above max Hz and below min Hz
            freq_mask = (frequencies <= max_fq) & (frequencies >= min_fq)
            frequencies = frequencies[freq_mask]

            spectrogram_db = spectrogram_db[freq_mask, :]

            db_mask = (spectrogram_db < min_db) | (spectrogram_db > max_db)

            # Mask spectrogram values above max dB and below min dB
            spectrogram_db[db_mask] = np.nan  # Optional: NaNs can appear as white in plot

            # Plot the spectrogram with filtered values
            plt.figure(figsize=(100, 6))
            plt.pcolormesh(times, frequencies, spectrogram_db, shading='gouraud', cmap=cm.binary) #.gray) #'inferno'
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.title('Spectrogram')
            plt.colorbar(label='Intensity [dB]')
            plt.savefig(png_path)
            
def fourier(input_directory,output_directory):
    # Read the WAV file
    for filename in os.listdir(input_directory):
        if filename.endswith(".wav"):

            wav_path = os.path.join(input_directory, filename)
            png_filename = os.path.splitext(filename)[0] + "_fourier.png"
            png_path = os.path.join(output_directory, png_filename)

            sample_rate, samples = wavfile.read(wav_path)

            # Compute the Fourier transform
            n = len(samples)
            freqs = np.fft.fftfreq(n, d=1/sample_rate)  # Frequency bins
            fft_values = np.fft.fft(samples)            # Fourier transform values

            # Plot the Fourier transform (magnitude)
            plt.figure(figsize=(10, 6))
            plt.plot(freqs[:n // 2], np.abs(fft_values[:n // 2]))  # Plot the positive frequencies only
            plt.title("Fourier Transform")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.savefig(png_path)
def cut(input_directory,output_directory,start_sec,end_sec):

    second = 1000
    for filename in os.listdir(input_directory):
        if filename.endswith(".wav"):

            wav_path = os.path.join(input_directory, filename)
            cuted_filename = os.path.splitext(filename)[0] + "_cuted.wav"
            cuted_path = os.path.join(output_directory, cuted_filename)

            song = AudioSegment.from_wav(wav_path)

            cuted_song = song[second*start_sec:second*end_sec] 

            cuted_song.export(cuted_filename, format="wav") 


def derivative(input_directory,output_directory):

    for filename in os.listdir(input_directory):
        if filename.endswith(".wav"):

            wav_path = os.path.join(input_directory, filename)
            derivative_filename = os.path.splitext(filename)[0] + "_derivative.wav"
            derivative_path = os.path.join(output_directory, derivative_filename)

            sample_rate, samples = wavfile.read(wav_path)

            # Compute the time derivative of the signal (numerical differentiation)
            derivative = np.diff(samples) / (1 / sample_rate)
            
            wavfile.write(derivative_path, sample_rate, samples)
