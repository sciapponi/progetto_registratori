import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import find_peaks, butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Create a Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply a Butterworth bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def compute_adaptive_parameters(y, sr, lowcut, highcut):
    """
    Compute adaptive parameters for bird call detection based on the overall filtered energy.
    Returns adaptive prominence for peak detection and an adaptive energy threshold.
    """
    # Filter the audio to the bird call frequency band.
    y_filtered = apply_bandpass_filter(y, lowcut, highcut, sr, order=4)
    
    # Compute a short-time amplitude envelope using a moving average.
    frame_length = int(sr * 0.05)  # 50 ms window
    hop_length = int(sr * 0.01)    # 10 ms hop
    envelope_frames = librosa.util.frame(np.abs(y_filtered), frame_length=frame_length, hop_length=hop_length)
    envelope = envelope_frames.mean(axis=0)
    
    # Use the median and MAD (median absolute deviation) as robust measures.
    median_env = np.median(envelope)
    mad_env = np.median(np.abs(envelope - median_env))
    
    # Set the adaptive prominence: median + k * MAD (tune k as needed)
    adaptive_prominence = median_env + 1.0 * mad_env

    # Compute overall RMS energy of the filtered signal.
    rms_all = np.sqrt(np.mean(y_filtered ** 2))
    # Set the adaptive energy threshold (this is the baseline, and will be lowered for background verification).
    adaptive_energy_threshold = 0.5 * rms_all

    print(f"Adaptive prominence set to {adaptive_prominence:.4f}")
    print(f"Adaptive energy threshold (baseline) set to {adaptive_energy_threshold:.4f}")

    return adaptive_prominence, adaptive_energy_threshold

def extract_call_segments(audio_path, output_folder, clip_duration=3.0, sr=22050,
                          lowcut=2000, highcut=10000):
    """
    Detects bird calls by applying a bandpass filter and using an adaptive threshold
    for peak detection, then extracts 3-second clips centered on detected peaks.
    Returns a list of call intervals as (start_time, end_time).
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Compute adaptive parameters for detection.
    adaptive_prominence, _ = compute_adaptive_parameters(y, sr, lowcut, highcut)
    
    # Filter the full audio for detection.
    y_filtered = apply_bandpass_filter(y, lowcut, highcut, sr, order=4)
    
    # Compute amplitude envelope from the filtered signal.
    frame_length = int(sr * 0.05)
    hop_length = int(sr * 0.01)
    envelope = librosa.util.frame(np.abs(y_filtered), frame_length=frame_length, hop_length=hop_length).mean(axis=0)
    
    # Detect peaks using the adaptive prominence.
    peaks, _ = find_peaks(envelope, prominence=adaptive_prominence)
    peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)

    call_intervals = []
    for t in peak_times:
        start_time = t - clip_duration / 2
        end_time = t + clip_duration / 2
        if start_time < 0 or end_time > duration:
            continue

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[start_sample:end_sample]

        filename = os.path.join(output_folder, f"call_{t:.2f}.wav")
        sf.write(filename, segment, sr)
        print(f"Saved call clip: {filename}")

        call_intervals.append((start_time, end_time))
    
    return call_intervals, y, sr, duration

def merge_intervals(intervals):
    """Merge overlapping or adjacent intervals."""
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], current[1]))
        else:
            merged.append(current)
    return merged

def identify_quietest_segments(y, sr, num_segments=5, clip_duration=3.0, lowcut=2000, highcut=10000):
    """
    Identifies the quietest segments in the audio based on the energy in the bird call frequency band.
    Returns a list of (start_time, end_time) intervals for the quietest segments.
    """
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    # Number of full segments we can extract
    num_possible_segments = int(total_duration // clip_duration)
    
    # Store energy for each segment
    segment_energies = []
    
    for i in range(num_possible_segments):
        start_time = i * clip_duration
        end_time = start_time + clip_duration
        
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[start_sample:end_sample]
        
        # Apply bandpass filter to focus on bird call frequencies
        filtered_segment = apply_bandpass_filter(segment, lowcut, highcut, sr)
        
        # Calculate energy in the bird call frequency band
        energy = np.sqrt(np.mean(filtered_segment ** 2))
        
        segment_energies.append((start_time, end_time, energy))
    
    # Sort segments by energy (lowest first)
    segment_energies.sort(key=lambda x: x[2])
    
    # Return the times of the quietest segments
    quietest_segments = [(s, e) for s, e, _ in segment_energies[:num_segments]]
    
    return quietest_segments

def extract_background_segments(audio_path, output_folder, call_intervals, clip_duration=3.0,
                               sr=22050, call_extension=0.5, lowcut=2000, highcut=10000,
                               num_background_clips=5):
    """
    New approach: Instead of trying to threshold, we'll just select the N quietest segments 
    that don't overlap with our detected bird calls.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    # Extend the call intervals by call_extension seconds on each side
    expanded_calls = [(max(0, s - call_extension), min(total_duration, e + call_extension))
                     for s, e in call_intervals]
    merged_calls = merge_intervals(expanded_calls)
    
    # Determine background intervals as the complement of the merged call intervals
    background_intervals = []
    prev_end = 0.0
    for s, e in merged_calls:
        if s > prev_end:
            background_intervals.append((prev_end, s))
        prev_end = e
    if prev_end < total_duration:
        background_intervals.append((prev_end, total_duration))
    
    # Extract all possible clip segments from background intervals
    candidate_segments = []
    for interval_start, interval_end in background_intervals:
        interval_length = interval_end - interval_start
        if interval_length < clip_duration:
            continue
        
        num_clips = int(interval_length // clip_duration)
        for i in range(num_clips):
            seg_start_time = interval_start + i * clip_duration
            seg_end_time = seg_start_time + clip_duration
            
            start_sample = int(seg_start_time * sr)
            end_sample = int(seg_end_time * sr)
            segment = y[start_sample:end_sample]
            
            # Calculate energy in the bird call frequency band
            filtered_segment = apply_bandpass_filter(segment, lowcut, highcut, sr)
            energy = np.sqrt(np.mean(filtered_segment ** 2))
            
            candidate_segments.append((seg_start_time, seg_end_time, energy))
    
    # Sort by energy and select the quietest ones
    candidate_segments.sort(key=lambda x: x[2])
    
    # Take at most num_background_clips or as many as we found if less
    num_to_extract = min(num_background_clips, len(candidate_segments))
    
    # Extract and save the quietest clips
    for i in range(num_to_extract):
        start_time, end_time, energy = candidate_segments[i]
        
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[start_sample:end_sample]
        
        filename = os.path.join(output_folder, f"background_{start_time:.2f}.wav")
        sf.write(filename, segment, sr)
        print(f"Saved background clip: {filename} (energy: {energy:.6f})")
    
    if num_to_extract == 0:
        print("No background clips were found. Try decreasing call_extension to allow more potential background segments.")

if __name__ == "__main__":
    # Set paths and parameters
    audio_path = "bird1m.mp3"  # Replace with your audio file path
    output_folder = "output_segments"
    os.makedirs(output_folder, exist_ok=True)
    
    # General parameters
    clip_duration = 1.0
    sr = 22050
    call_extension = 0.5  # Extend call intervals to avoid bleed-over
    
    # Frequency range for bird calls (adjust based on species)
    lowcut = 2000    # Lower bound (Hz)
    highcut = 10000  # Upper bound (Hz)
    
    # Number of background clips to extract
    num_background_clips = 5
    
    # Load the audio once to compute adaptive parameters
    y, sr = librosa.load(audio_path, sr=sr)
    adaptive_prominence, adaptive_energy_threshold = compute_adaptive_parameters(y, sr, lowcut, highcut)
    
    # Extract bird call segments using adaptive prominence
    call_intervals, y, sr, duration = extract_call_segments(
        audio_path, output_folder, clip_duration, sr, lowcut, highcut
    )
    
    # Use the new approach to extract background segments
    extract_background_segments(
        audio_path, output_folder, call_intervals, clip_duration, sr,
        call_extension, lowcut, highcut, num_background_clips
    )