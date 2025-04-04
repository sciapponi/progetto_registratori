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
    adaptive_prominence = median_env + 1.5 * mad_env  # Increased from 1.0 to 1.5 for more selective detection

    # Compute overall RMS energy of the filtered signal.
    rms_all = np.sqrt(np.mean(y_filtered ** 2))
    # Set the adaptive energy threshold (this is the baseline, and will be lowered for background verification).
    adaptive_energy_threshold = 0.5 * rms_all

    print(f"Adaptive prominence set to {adaptive_prominence:.4f}")
    print(f"Adaptive energy threshold (baseline) set to {adaptive_energy_threshold:.4f}")

    return adaptive_prominence, adaptive_energy_threshold

def extract_call_segments(audio_path, output_folder, clip_duration=3.0, sr=22050,
                          lowcut=2000, highcut=10000, min_peak_distance=1.0, height_percentile=75):
    """
    Detects bird calls by applying a bandpass filter and using an adaptive threshold
    for peak detection, then extracts clips centered on detected peaks.
    Added parameters to reduce overlapping calls:
    - min_peak_distance: Minimum time between peaks in seconds
    - height_percentile: Only keep peaks above this percentile
    
    Returns a list of call intervals as (start_time, end_time), along with the audio data, sample rate, and duration.
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
    envelope_frames = librosa.util.frame(np.abs(y_filtered), frame_length=frame_length, hop_length=hop_length)
    envelope = envelope_frames.mean(axis=0)
    
    # Calculate minimum distance between peaks in frames
    min_peak_distance_frames = int(min_peak_distance / (hop_length / sr))
    
    # Detect peaks using the adaptive prominence and minimum distance
    peaks, properties = find_peaks(envelope, 
                                  prominence=adaptive_prominence,
                                  distance=min_peak_distance_frames)
    
    # Handle case with no peaks detected
    if len(peaks) == 0:
        print("No peaks detected. Try adjusting detection parameters.")
        return [], y, sr, duration
    
    # Sort peaks by prominence (highest first)
    sorted_indices = np.argsort(-properties['prominences'])
    sorted_peaks = peaks[sorted_indices]
    sorted_prominences = properties['prominences'][sorted_indices]
    
    # Keep only the top percentile of peaks based on height/amplitude
    if len(sorted_peaks) > 0:  # Check if any peaks were found
        height_threshold = np.percentile(envelope[sorted_peaks], height_percentile)
        selected_peaks = [p for i, p in enumerate(sorted_peaks) if envelope[p] >= height_threshold]
    else:
        selected_peaks = []
    
    # Convert peaks to time
    peak_times = librosa.frames_to_time(selected_peaks, sr=sr, hop_length=hop_length)
    
    print(f"Detected {len(peak_times)} significant bird calls")

    call_intervals = []
    for i, t in enumerate(peak_times):
        start_time = max(0, t - clip_duration / 2)  # Ensure we don't go below 0
        end_time = min(duration, t + clip_duration / 2)  # Ensure we don't exceed audio length

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[start_sample:end_sample]

        filename = os.path.join(output_folder, f"call_{i+1:03d}_{t:.2f}.wav")
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

def extract_background_segments_with_concatenation(audio_path, output_folder, call_intervals, 
                                                  desired_clip_duration=3.0, min_segment_length=0.2,
                                                  sr=22050, call_extension=0.5, lowcut=2000, highcut=10000,
                                                  num_background_clips=5):
    """
    Extracts background segments by:
    1. Finding non-call regions
    2. Breaking them into smaller chunks
    3. Calculating energy for each chunk
    4. Selecting the quietest chunks
    5. Concatenating them to reach the desired duration
    """
    y, sr = librosa.load(audio_path, sr=sr)
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    # Handle case with no call intervals
    if not call_intervals:
        print("No call intervals provided. Extracting background from quietest regions.")
        # Get energy profile of entire audio
        frame_length = int(sr * min_segment_length)
        hop_length = int(frame_length / 2)
        
        # Compute energy in chunks across the entire audio
        n_frames = 1 + (len(y) - frame_length) // hop_length
        frames = []
        energies = []
        
        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            if end <= len(y):
                frame = y[start:end]
                # Calculate energy in the bird call frequency band
                filtered_frame = apply_bandpass_filter(frame, lowcut, highcut, sr)
                energy = np.sqrt(np.mean(filtered_frame ** 2))
                frames.append((start/sr, end/sr))
                energies.append(energy)
        
        # Sort frames by energy
        sorted_indices = np.argsort(energies)
        
        # Extract lowest energy segments
        for bg_idx in range(min(num_background_clips, len(sorted_indices))):
            idx = sorted_indices[bg_idx]
            start_time, end_time = frames[idx]
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment = y[start_sample:end_sample]
            
            # Extend segment if it's too short
            while len(segment) < desired_clip_duration * sr and bg_idx + 1 < len(sorted_indices):
                bg_idx += 1
                next_idx = sorted_indices[bg_idx]
                next_start, next_end = frames[next_idx]
                next_start_sample = int(next_start * sr)
                next_end_sample = int(next_end * sr)
                next_segment = y[next_start_sample:next_end_sample]
                segment = np.concatenate([segment, next_segment])
            
            # Trim to desired length
            if len(segment) > desired_clip_duration * sr:
                segment = segment[:int(desired_clip_duration * sr)]
            
            filename = os.path.join(output_folder, f"background_{bg_idx+1:03d}.wav")
            sf.write(filename, segment, sr)
            print(f"Saved background clip: {filename}")
        
        return
    
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
    
    if not background_intervals:
        print("No background intervals found after merging calls.")
        return
    
    # Extract all possible smaller segments from background intervals
    candidate_segments = []
    
    # Minimum number of samples for a segment to be considered
    min_segment_samples = int(min_segment_length * sr)
    
    for interval_start, interval_end in background_intervals:
        interval_length = interval_end - interval_start
        if interval_length < min_segment_length:
            continue
        
        # Calculate how many chunks we can extract from this interval
        # Use a small step size to get more candidates
        step_size = min_segment_length / 2  # Smaller step size for more overlap
        
        current_pos = interval_start
        while current_pos + min_segment_length <= interval_end:
            # Determine the end of this segment (either min length or what's available)
            segment_end = min(current_pos + min_segment_length, interval_end)
            segment_duration = segment_end - current_pos
            
            if segment_duration >= min_segment_length:
                start_sample = int(current_pos * sr)
                end_sample = int(segment_end * sr)
                
                if end_sample - start_sample >= min_segment_samples:
                    segment = y[start_sample:end_sample]
                    
                    # Calculate energy in the bird call frequency band
                    filtered_segment = apply_bandpass_filter(segment, lowcut, highcut, sr)
                    energy = np.sqrt(np.mean(filtered_segment ** 2))
                    
                    candidate_segments.append((current_pos, segment_end, energy, segment))
            
            current_pos += step_size
    
    if not candidate_segments:
        print("No background segments found. Try reducing call_extension or min_segment_length.")
        return
    
    # Sort by energy (lowest first)
    candidate_segments.sort(key=lambda x: x[2])
    
    # Create concatenated background clips
    for bg_clip_num in range(num_background_clips):
        # Reset for each desired output clip
        concatenated_audio = np.array([])
        used_segments = []
        current_duration = 0
        
        # Keep adding segments until we reach the desired duration
        segment_index = 0
        segments_added = 0
        
        while current_duration < desired_clip_duration and segment_index < len(candidate_segments):
            # Skip segments we've already used for this clip
            if segment_index in used_segments:
                segment_index += 1
                continue
                
            # Get the next quietest segment
            _, _, energy, segment = candidate_segments[segment_index]
            
            # Concatenate the segment with a small fade in/out to avoid clicks
            if len(concatenated_audio) > 0 and len(segment) > 0:
                # Apply a short crossfade (10ms)
                fade_length = min(int(0.01 * sr), len(segment), len(concatenated_audio))
                fade_in = np.linspace(0, 1, fade_length)
                fade_out = np.linspace(1, 0, fade_length)
                
                segment[:fade_length] *= fade_in
                concatenated_audio[-fade_length:] *= fade_out
            
            concatenated_audio = np.concatenate([concatenated_audio, segment])
            
            # Update tracking variables
            used_segments.append(segment_index)
            current_duration += len(segment) / sr
            segment_index += 1
            segments_added += 1
            
            # If we've exhausted all segments, break
            if segment_index >= len(candidate_segments):
                break
        
        # If we couldn't add any segments, we're done
        if segments_added == 0:
            print(f"Could not create background clip {bg_clip_num+1}. No more available segments.")
            break
            
        # If we have more audio than needed, trim it
        desired_length_samples = int(desired_clip_duration * sr)
        if len(concatenated_audio) > desired_length_samples:
            # Apply a fade out at the end
            fade_length = min(int(0.05 * sr), len(concatenated_audio))
            fade_out = np.linspace(1, 0, fade_length)
            concatenated_audio[-fade_length:] *= fade_out
            
            # Trim to the exact length needed
            concatenated_audio = concatenated_audio[:desired_length_samples]
            
        # Save the concatenated clip
        filename = os.path.join(output_folder, f"background_{bg_clip_num+1:03d}.wav")
        sf.write(filename, concatenated_audio, sr)
        print(f"Saved concatenated background clip {bg_clip_num+1}: {filename} ({current_duration:.2f}s from {segments_added} segments)")
        
        # If we've used all segments, no need to try making more clips
        if len(used_segments) >= len(candidate_segments):
            print("Used all available background segments.")
            break

def main(audio_path, output_folder="output_segments", clip_duration=3.0, 
         desired_background_duration=3.0, min_segment_length=0.2, sr=22050,
         call_extension=0.5, lowcut=2000, highcut=10000, min_peak_distance=1.5,
         height_percentile=75, num_background_clips=5):
    """
    Main function to extract bird calls and background segments from an audio file.
    
    Parameters:
    -----------
    audio_path : str
        Path to the input audio file
    output_folder : str, optional
        Directory to save extracted segments
    clip_duration : float, optional
        Duration of bird call clips in seconds
    desired_background_duration : float, optional
        Desired duration for background clips in seconds
    min_segment_length : float, optional
        Minimum length for background segments to consider in seconds
    sr : int, optional
        Sample rate for audio processing
    call_extension : float, optional
        Extend call intervals by this amount in seconds to avoid bleed-over
    lowcut : int, optional
        Lower bound frequency (Hz) for bird call detection
    highcut : int, optional
        Upper bound frequency (Hz) for bird call detection
    min_peak_distance : float, optional
        Minimum time between detected calls in seconds
    height_percentile : int, optional
        Only keep peaks above this percentile (0-100)
    num_background_clips : int, optional
        Number of background clips to extract
    """
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Processing {audio_path}...")
    print(f"Parameters: clip_duration={clip_duration}s, background_duration={desired_background_duration}s")
    print(f"Frequency band: {lowcut}-{highcut} Hz")
    
    # Extract bird call segments
    call_intervals, y, sr, duration = extract_call_segments(
        audio_path, output_folder, clip_duration, sr, lowcut, highcut,
        min_peak_distance, height_percentile
    )
    
    print(f"Audio duration: {duration:.2f}s")
    print(f"Extracted {len(call_intervals)} call segments")
    
    # Extract background segments
    extract_background_segments_with_concatenation(
        audio_path, output_folder, call_intervals, 
        desired_background_duration, min_segment_length,
        sr, call_extension, lowcut, highcut, num_background_clips
    )
    
    print("Processing complete!")

if __name__ == "__main__":
    # Set paths and parameters
    audio_path = "chirping.mp3"  # Replace with your audio file path
    output_folder = "output_segments"
    
    # Call the main function with default parameters
    main(audio_path, output_folder)