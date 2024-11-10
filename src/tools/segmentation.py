import librosa
import scipy.signal as sg
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as io
from scipy.ndimage import label
import os
import soundfile as sf

def sos_filter(sample,sr):
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html#scipy.signal.butter

    sos = sg.butter(4, (1000,8000), btype='band', output='sos', fs=sr)
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html#scipy.signal.sosfilt

    return sg.sosfilt(sos,sample)


def ba_filter(sample,sr):
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html#scipy.signal.butter

    (b,a) = sg.butter(4, (1000,8000), btype='band', output='ba', fs=sr)    
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html#scipy.signal.lfilter

    return sg.lfilter(b, a, sample)

def segment_signal(signal,sr,silence_threshold=0.5):
    # Segments the signal based on the silence_threshold; 
    # if silence exceeds the threshold, it segments before the silence and 
    # continues searching for the next segment

    silence_threshold = int(silence_threshold*sr) # Convert threshold from seconds to sample count
    segments = []
    current_segment = None
    silence_counter = 0

    for i, sample in enumerate(signal):
        if sample > 0:
            if current_segment is None:
                current_segment = [i, None]
            silence_counter = 0
        elif current_segment is not None:
            silence_counter += 1
            if silence_counter >= silence_threshold:
                current_segment[1] = i - silence_threshold
                segments.append(np.array(current_segment))
                current_segment = None
                silence_counter = 0

    if current_segment is not None:
        current_segment[1] = len(signal) - 1
        segments.append(np.array(current_segment))

    return np.array(segments)

def normalize_segments(segments,sr, target_length=3):
    target_length = int(target_length*sr) # Convert threshold from seconds to sample count

    normalized_segments = []

    for segment in segments:
        start, end = segment
        current_length = end - start + 1

        # If the segment length is already the target length, add the segment as is
        if current_length == target_length:
            normalized_segments.append((start, end))
        
        # If the segment is too short, extend it from the beginning and the end
        elif current_length < target_length:
            extra_length = target_length - current_length
            expand_start = extra_length // 2
            expand_end = extra_length - expand_start
            new_start = start - expand_start
            new_end = end + expand_end
            normalized_segments.append((new_start, new_end))

        # If the segment is too long, shorten it from the beginning and the end
        else:
            extra_length = current_length - target_length
            trim_start = extra_length // 2
            trim_end = extra_length - trim_start
            new_start = start + trim_start
            new_end = end - trim_end
            normalized_segments.append((new_start, new_end))

    return np.array(normalized_segments)

def save_segments_as_wav(signal, sr, output_directory, segments):

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Save each segment as a separate .wav file
    for i, (start, end) in enumerate(segments):
        segment_signal = signal[start:end + 1]
        segment_path = f"{output_directory}/segment_{i + 1}.wav"
        sf.write(segment_path, segment_signal, sr)

def segmentate(output_directory,sample,sr,theta=0.28,silence_threshold=0.5,target_length=3):

    sample = sos_filter(sample,sr)
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html#scipy.signal.hilbert
    h_sample = sg.hilbert(sample)
    h_conj = np.conjugate(h_sample)
    y_sample = np.sqrt(h_sample*h_conj)

    activity = y_sample > theta
    segments = segment_signal(activity,sr,silence_threshold=silence_threshold)
    std_segments = normalize_segments(segments,sr,target_length=target_length)

    save_segments_as_wav(sample,sr,output_directory,std_segments)

def segmentate_all(input_directory,output_directory,theta=0.28,silence_threshold=0.5,target_length=3):
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Iterate over all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".wav"):
            wav_path = os.path.join(input_directory, filename)
            seg_output_directory = os.path.splitext(filename)[0]
            seg_output_path = os.path.join(output_directory, seg_output_directory)
            
            sample,sr = librosa.load(wav_path,mono=True,sr=32000)
            segmentate(seg_output_path,sample,sr,theta,silence_threshold,target_length)