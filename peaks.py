import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Function for peak detection and visualization
def detect_and_plot_peaks(audio_path, prominence=0.1):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)  # Use the native sampling rate of the audio

    # Compute the onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # Detect peaks using the onset envelope
    peaks = librosa.util.peak_pick(onset_env,
                                   pre_max=15,  # Maximum number of frames before peak
                                   post_max=15, # Maximum number of frames after peak
                                   pre_avg=3,  # Average frames before peak
                                   post_avg=3, # Average frames after peak
                                   delta=prominence, # Minimum peak height
                                   wait=5)    # Minimum frames between peaks

    # Convert peak positions to time
    peak_times = librosa.frames_to_time(peaks, sr=sr)

    # Plot the waveform with peaks highlighted
    plt.figure(figsize=(14, 6))

    # Waveform plot
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Highlight the peaks on the waveform
    for peak_time in peak_times:
        plt.axvline(x=peak_time, color='r', linestyle='--', alpha=0.8, label='Peak')

    plt.legend(['Waveform', 'Peak'])

    # Onset envelope plot
    plt.subplot(2, 1, 2)
    times = librosa.times_like(onset_env, sr=sr)
    plt.plot(times, onset_env, label='Onset Envelope')
    plt.vlines(peak_times, ymin=0, ymax=max(onset_env), color='r', linestyle='--', alpha=0.8, label='Peaks')
    plt.title('Onset Envelope and Peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Onset Strength')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage
# Replace 'path_to_audio_file.wav' with the path to your audio file
detect_and_plot_peaks('/home/ste/Code/birds/dataset/Phoenicurus_ochruros/BlackRedstart17Feb2009TafraouteMorocco.mp3', prominence=0.7)
detect_and_plot_peaks("/home/ste/Code/birds/dataset/Phoenicurus_ochruros/codirossospazzacamino7.mp3", prominence=0.9)