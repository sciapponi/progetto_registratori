from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence
import numpy as np
import scipy.signal as signal
import librosa
import soundfile as sf
import os
import matplotlib.pyplot as plt

def plot_audio_comparison(original_audio, original_sr, processed_audio, processed_sr, title="Audio Processing Results"):
    """
    Plot waveforms and spectrograms of original and processed audio.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title)
    
    # Plot original waveform
    time_orig = np.arange(len(original_audio)) / original_sr
    axes[0, 0].plot(time_orig, original_audio)
    axes[0, 0].set_title('Original Waveform')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    
    # Plot processed waveform
    time_proc = np.arange(len(processed_audio)) / processed_sr
    axes[0, 1].plot(time_proc, processed_audio)
    axes[0, 1].set_title('Processed Waveform')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    
    # Plot original spectrogram
    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max)
    librosa.display.specshow(D_orig, y_axis='log', x_axis='time', ax=axes[1, 0])
    axes[1, 0].set_title('Original Spectrogram')
    
    # Plot processed spectrogram
    D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(processed_audio)), ref=np.max)
    img = librosa.display.specshow(D_proc, y_axis='log', x_axis='time', ax=axes[1, 1])
    axes[1, 1].set_title('Processed Spectrogram')
    
    # Add colorbar
    fig.colorbar(img, ax=axes[1, :], format='%+2.0f dB')
    
    plt.tight_layout()
    return fig

def remove_silence_between_calls(input_file, output_file,
                               min_silence_len=1000,
                               silence_thresh=-40,
                               keep_silence=100,
                               low_freq=500,
                               high_freq=8000,
                               enable_noise_reduction=True,
                               plot=True):
    """
    Remove silence between bird calls with noise reduction and frequency filtering.
    Optionally plots the results.
    """
    # Load audio file using librosa
    original_audio, sr = librosa.load(input_file, sr=None)
    
    # Apply bandpass filter
    audio_filtered = bandpass_filter(original_audio, sr, low_freq, high_freq)
    
    # Apply noise reduction if enabled
    if enable_noise_reduction:
        audio_filtered = reduce_noise(audio_filtered, sr)
    
    # Convert to pydub format for silence removal
    temp_file = "temp_filtered.wav"
    sf.write(temp_file, audio_filtered, sr)
    audio = AudioSegment.from_wav(temp_file)
    os.remove(temp_file)
    
    # Split audio into non-silent chunks
    audio_chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
    ranges = detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )
    print(ranges) 
    # exit()
    
    # Combine chunks with minimal silence
    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
    
    # Export the result
    combined.export(output_file, format=os.path.splitext(output_file)[1][1:])
    
    # Load processed audio for plotting
    processed_audio, processed_sr = librosa.load(output_file, sr=None)
    
    if plot:
        fig = plot_audio_comparison(original_audio, sr, processed_audio, processed_sr)
        plt.show()
    
    return len(audio_chunks)

# Keep the existing noise reduction and bandpass filter functions...
def reduce_noise(audio_data, sr, noise_clip=None):
    """
    Reduce background noise using spectral gating.
    
    Parameters:
    - audio_data: numpy array of audio samples
    - sr: sampling rate
    - noise_clip: optional noise profile to use as reference
    """
    # If no noise profile provided, estimate from the quietest section
    if noise_clip is None:
        frame_length = int(sr * 0.025)  # 25ms frames
        hop_length = int(sr * 0.010)    # 10ms hop
        
        # Calculate frame energies
        frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)
        frame_energies = np.sum(frames**2, axis=0)
        
        # Get the quietest frames as noise reference
        quietest_frame_idx = np.argsort(frame_energies)[:int(len(frame_energies) * 0.1)]
        noise_clip = np.concatenate([frames[:, i] for i in quietest_frame_idx])

    # Compute noise profile
    noise_stft = librosa.stft(noise_clip)
    noise_power = np.mean(np.abs(noise_stft)**2, axis=1)
    
    # Compute STFT of input signal
    signal_stft = librosa.stft(audio_data)
    signal_power = np.abs(signal_stft)**2
    
    # Compute threshold for spectral gating
    threshold = 2.0 * noise_power[:, None]
    
    # Apply spectral gate
    mask = signal_power > threshold
    signal_stft_cleaned = signal_stft * mask
    
    # Reconstruct signal
    audio_cleaned = librosa.istft(signal_stft_cleaned)
    return audio_cleaned

def bandpass_filter(audio_data, sr, low_freq=150, high_freq=8000):
    """
    Apply bandpass filter to focus on typical bird call frequencies.
    """
    nyquist = sr // 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, audio_data)

# Example usage
if __name__ == "__main__":
    # input_path = "dataset/Hirundo_rustica/BarnSwallow.mp3"
    input_path = "/home/ste/Code/progetto_registratori/dataset/Phoenicurus_ochruros/BlackRedstart17Feb2009TafraouteMorocco.mp3"
    input_path = "/home/ste/Code/progetto_registratori/dataset/Phoenicurus_ochruros/codirossospazzacamino7.mp3"
    output_path = "processed_bird_calls.wav"
    
    num_calls = remove_silence_between_calls(
        input_path,
        output_path,
        min_silence_len=1000,
        silence_thresh=-40,
        keep_silence=100,
        low_freq=500,
        high_freq=8000,
        enable_noise_reduction=True,
        plot=True  # This will show the visualization
    )
    
    print(f"Processed {num_calls} bird calls")
