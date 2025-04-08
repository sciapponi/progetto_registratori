import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import find_peaks, butter, filtfilt
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import audioread
from scipy import signal


# Existing filter functions remain the same
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
    adaptive_prominence = median_env + 1.5 * mad_env

    # Compute overall RMS energy of the filtered signal.
    rms_all = np.sqrt(np.mean(y_filtered ** 2))
    # Set the adaptive energy threshold (this is the baseline, and will be lowered for background verification).
    adaptive_energy_threshold = 0.5 * rms_all

    return adaptive_prominence, adaptive_energy_threshold

def extract_call_segments(audio_path, output_folder, clip_duration=3.0, sr=22050,
                          lowcut=2000, highcut=10000, min_peak_distance=1.0, height_percentile=75,
                          verbose=True):
    """
    Detects bird calls by applying a bandpass filter and using an adaptive threshold
    for peak detection, then extracts clips centered on detected peaks.
    
    Returns a list of call intervals as (start_time, end_time), along with the audio data, sample rate, and duration.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
     # Check if file exists
    if not os.path.exists(audio_path):
        if verbose:
            print(f"File not found: {audio_path}")
        return [], None, None, 0
    
    try:
        # Load audio with downsampling if needed
        y, sr = librosa.load(audio_path, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)
    
        # Get the filename without extension for naming saved clips
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        
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
            if verbose:
                print(f"No peaks detected in {audio_path}. Try adjusting detection parameters.")
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
        
        if verbose:
            print(f"Detected {len(peak_times)} significant bird calls in {audio_path}")

        call_intervals = []
        for i, t in enumerate(peak_times):
            start_time = max(0, t - clip_duration / 2)  # Ensure we don't go below 0
            end_time = min(duration, t + clip_duration / 2)  # Ensure we don't exceed audio length

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment = y[start_sample:end_sample]

            # Include base filename in the output filename to identify source
            filename = os.path.join(output_folder, f"{base_filename}_call_{i+1:03d}.wav")
            sf.write(filename, segment, sr)
            if verbose:
                print(f"Saved call clip: {filename}")

            call_intervals.append((start_time, end_time))
    except audioread.exceptions.NoBackendError:
        if verbose:
            print(f"Error: No audio backend found for processing {audio_path}. Make sure ffmpeg is installed.")
        return [], None, None, 0
    except Exception as e:
        if verbose:
            print(f"Error processing {audio_path}: {str(e)}")
        return [], None, None, 0    
    
    return call_intervals, y, sr, duration

def extract_background_segments(audio_path, output_folder, call_intervals,
                           desired_clip_duration=3.0, sr=22050,
                           call_extension=1.0, num_clips=5,
                           lowcut=2000, highcut=10000,
                           energy_threshold=3.0, min_flatness=0.5,
                           band_energy_ratio_threshold=2.0,
                           verbose=True):
    """
    Enhanced background extraction with multi-feature analysis
    """
    # Safe output folder creation
    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        if verbose:
            print(f"Could not create output folder: {e}")
        return 0

    # Validate inputs
    if not os.path.exists(audio_path):
        if verbose:
            print(f"Audio file not found: {audio_path}")
        return 0

    # Load audio with error handling
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        if verbose:
            print(f"Error loading audio: {e}")
        return 0

    if len(y) == 0:
        if verbose:
            print("Empty audio file")
        return 0

    duration = len(y) / sr
    window_size = int(desired_clip_duration * sr)
    
    # Create extended call mask - avoid all known bird calls with extra padding
    background_mask = np.ones(len(y), dtype=bool)
    try:
        for start, end in call_intervals:
            # Add extra padding to ensure we avoid call onset/offset
            start_sample = max(0, int((start - call_extension * 1.5) * sr))
            end_sample = min(len(y), int((end + call_extension * 1.5) * sr))
            background_mask[start_sample:end_sample] = False
    except Exception as e:
        if verbose:
            print(f"Error processing call intervals: {e}")
        background_mask[:] = True  # Fallback to full audio

    # Calculate the bird call frequency band energy
    y_bird_band = apply_bandpass_filter(y, lowcut, highcut, sr)
    y_bird_band_energy = np.square(y_bird_band)
    
    # Calculate full spectrum energy for comparison
    y_energy = np.square(y)
    
    # Calculate energy ratio across the signal
    # Higher ratio means more energy in bird frequencies
    energy_ratio = np.zeros_like(y, dtype=float)
    
    # Calculate ratio using rolling windows for efficiency
    frame_length = int(0.05 * sr)  # 50ms frames
    hop_length = int(0.01 * sr)    # 10ms hop
    
    # Get energy in bird band vs total energy ratio
    bird_energy_frames = librosa.feature.rms(y=y_bird_band, frame_length=frame_length, hop_length=hop_length)[0]
    total_energy_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Prevent division by zero
    total_energy_frames = np.maximum(total_energy_frames, 1e-10)
    energy_ratio_frames = bird_energy_frames / total_energy_frames
    
    # Stretch energy ratio to match audio length
    frame_times = librosa.frames_to_time(np.arange(len(energy_ratio_frames)), sr=sr, hop_length=hop_length)
    sample_times = np.arange(len(y)) / sr
    energy_ratio = np.interp(sample_times, frame_times, energy_ratio_frames)
    
    # Mark locations with high bird-band energy as potential calls
    bird_energy_mask = energy_ratio < band_energy_ratio_threshold
    
    # Compute spectral flatness for background characterization
    try:
        n_fft = 1024  # Lower resolution for faster processing
        hop_length = 512
        
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        
        # Normalize flatness to 0-1 range
        flatness = (flatness - np.min(flatness)) / (np.max(flatness) - np.min(flatness) + 1e-10)
        
        # Interpolate to match audio length
        frame_times = librosa.frames_to_time(np.arange(len(flatness)), sr=sr, hop_length=hop_length)
        flatness_full = np.interp(sample_times, frame_times, flatness)
    except Exception as e:
        if verbose:
            print(f"Spectral analysis failed: {e}")
        flatness_full = np.ones(len(y)) * 0.5

    # Compute zero crossing rate - bird calls have higher ZCR
    try:
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
        zcr_times = librosa.frames_to_time(np.arange(len(zcr)), sr=sr, hop_length=hop_length)
        zcr_full = np.interp(sample_times, zcr_times, zcr)
        
        # Normalize ZCR
        zcr_full = (zcr_full - np.min(zcr_full)) / (np.max(zcr_full) - np.min(zcr_full) + 1e-10)
        
        # Low ZCR is better for background (less harmonic content)
        zcr_mask = zcr_full < 0.5
    except Exception as e:
        if verbose:
            print(f"ZCR analysis failed: {e}")
        zcr_mask = np.ones(len(y), dtype=bool)
    
    # Combine all masks for better background detection
    # Areas with high flatness, low bird-band energy, low ZCR, and away from known calls
    combined_mask = background_mask & bird_energy_mask & zcr_mask
    combined_quality = flatness_full * combined_mask
    
    # Apply minimum quality threshold 
    combined_quality[combined_quality < min_flatness] = 0
    
    # Find the best candidates using sliding window
    stride = max(1, window_size // 10)
    conv_size = len(combined_quality) - window_size + 1
    
    if conv_size <= 0:
        if verbose:
            print("Audio too short for background extraction")
        return 0
        
    # Sample at regular intervals for efficiency
    sample_points = np.arange(0, conv_size, stride)
    if len(sample_points) == 0:
        sample_points = [0]
        
    # Calculate mean quality values at each sample point
    sample_values = np.array([np.mean(combined_quality[i:i+window_size]) for i in sample_points])
    
    # Get top candidates (twice what we need for filtering)
    candidates = sample_points[np.argsort(-sample_values)[:num_clips*2]]

    # Final verification of candidates to ensure clean background
    clean_clips = []
    candidate_desc = tqdm(candidates, desc="Verifying candidates") if verbose else candidates
    
    for pos in candidate_desc:
        try:
            if pos + window_size > len(y):
                continue

            clip = y[pos:pos+window_size]
            
            # Multi-feature verification
            
            # 1. Check energy stability (no sudden peaks)
            clip_energy = librosa.feature.rms(y=clip, frame_length=1024, hop_length=512)[0]
            if len(clip_energy) < 3:
                continue
                
            # Reject if max energy is much higher than median (indicates a call)
            median_energy = np.median(clip_energy)
            if np.max(clip_energy) > median_energy * energy_threshold:
                continue
            
            # 2. Check bird band energy specifically
            clip_bird = apply_bandpass_filter(clip, lowcut, highcut, sr)
            bird_energy = np.sqrt(np.mean(clip_bird ** 2))
            full_energy = np.sqrt(np.mean(clip ** 2))
            
            # Skip if too much energy is in the bird frequency range
            if full_energy > 0 and bird_energy / full_energy > 0.3:
                continue
            
            # 3. Check spectrum peaks in bird range
            n_fft_fine = 2048  # Higher resolution for verification
            clip_spec = np.abs(librosa.stft(clip, n_fft=n_fft_fine))
            
            # Get frequency bin indices for bird range
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft_fine)
            bird_range_indices = np.where((freqs >= lowcut) & (freqs <= highcut))[0]
            
            if len(bird_range_indices) > 0:
                # Get max energy in bird range per time frame
                bird_range_energy = np.max(clip_spec[bird_range_indices, :], axis=0)
                # Get max energy in full spectrum
                full_range_energy = np.max(clip_spec, axis=0)
                
                # Calculate ratio of bird range energy to full spectrum
                bird_energy_ratio = np.mean(bird_range_energy / (full_range_energy + 1e-10))
                
                # Skip if too much energy in bird frequency range
                if bird_energy_ratio > 0.5:  
                    continue
            
            # Apply fade in/out to avoid clicks
            fade_len = min(512, window_size // 10)
            clip[:fade_len] *= np.linspace(0, 1, fade_len)
            clip[-fade_len:] *= np.linspace(1, 0, fade_len)
            
            clean_clips.append((clip, pos))
            
            if len(clean_clips) >= num_clips:
                break
        except Exception as e:
            if verbose:
                print(f"Error processing candidate: {e}")
            continue

    # Save valid clips
    saved_count = 0
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]
    
    for i, (clip, _) in enumerate(clean_clips):
        try:
            output_path = os.path.join(output_folder, f"{base_filename}_bg_{i}.wav")
            sf.write(output_path, clip, sr)
            saved_count += 1
        except Exception as e:
            if verbose:
                print(f"Error saving clip {i}: {e}")

    if verbose:
        print(f"Successfully saved {saved_count}/{num_clips} background clips")
    return saved_count

# def extract_background_segments(audio_path, output_folder, call_intervals,
#                               desired_clip_duration=3.0, sr=22050,
#                               call_extension=1.5,  # Increased buffer around calls
#                               num_clips=5,
#                               lowcut=2000, highcut=10000,
#                               min_energy=0.001, max_energy=0.1,
#                               mode='hybrid', 
#                               noise_reduction_strength=0.85,
#                               verbose=True):
#     """
#     Enhanced background extraction with aggressive bird call removal.
    
#     Parameters:
#     -----------
#     noise_reduction_strength : float (0-1)
#         How aggressively to remove non-stationary components (higher = more removal)
#     """
#     import os
#     import numpy as np
#     import librosa
#     import soundfile as sf
#     import noisereduce as nr
#     from scipy import signal

#     # Setup
#     os.makedirs(output_folder, exist_ok=True)
#     base_name = os.path.splitext(os.path.basename(audio_path))[0]
#     saved_count = 0

#     # Load audio with extended buffer for processing
#     y, sr = librosa.load(audio_path, sr=sr, mono=True)
#     duration = len(y) / sr

#     # Create extended exclusion mask (wider buffer around calls)
#     exclusion_mask = np.ones(len(y), dtype=bool)
#     for start, end in call_intervals:
#         start_sample = max(0, int((start - call_extension*1.5) * sr))  # Extra buffer
#         end_sample = min(len(y), int((end + call_extension*1.5) * sr))
#         exclusion_mask[start_sample:end_sample] = False

#     # =============================================
#     # STAGE 1: Find clean segments between calls
#     # =============================================
#     if mode in ('real', 'hybrid'):
#         segment_samples = int(desired_clip_duration * sr)
        
#         # Find all candidate segments (longer than target duration)
#         valid_segments = []
#         in_segment = False
#         segment_start = 0
        
#         for i in range(len(exclusion_mask)):
#             if exclusion_mask[i] and not in_segment:
#                 segment_start = i
#                 in_segment = True
#             elif not exclusion_mask[i] and in_segment:
#                 if (i - segment_start) >= segment_samples:
#                     valid_segments.append((segment_start, i))
#                 in_segment = False
        
#         # Process candidates with aggressive noise cleaning
#         clean_segments = []
#         for start, end in valid_segments[:num_clips*3]:  # Process extra candidates
#             segment = y[start:end]
            
#             # STAGE 2: Spectral gating to remove residual calls
#             # Use the first 500ms as noise profile
#             noise_profile = segment[:int(0.5*sr)]
            
#             # Aggressive reduction
#             clean_segment = nr.reduce_noise(
#                 y=segment,
#                 y_noise=noise_profile,
#                 sr=sr,
#                 stationary=True,
#                 prop_decrease=noise_reduction_strength,
#                 freq_mask_smooth_hz=500,
#                 time_mask_smooth_ms=500,
#                 n_fft=2048,
#                 win_length=2048
#             )
            
#             # STAGE 3: Bandpass and energy validation
#             # Remove remaining bird band energy
#             nyquist = 0.5 * sr
#             low = lowcut / nyquist
#             high = highcut / nyquist
#             b, a = signal.butter(5, [low, high], btype='bandstop')
#             filtered = signal.filtfilt(b, a, clean_segment)
            
#             # Energy check
#             rms = np.sqrt(np.mean(filtered**2))
#             if min_energy <= rms <= max_energy:
#                 clean_segments.append(filtered)
#                 if len(clean_segments) >= num_clips:
#                     break

#         # Save clean segments
#         for i, seg in enumerate(clean_segments[:num_clips]):
#             output_path = os.path.join(output_folder, f"{base_name}_bg_clean_{i}.wav")
#             sf.write(output_path, seg, sr)
#             saved_count += 1

#     # =============================================
#     # STAGE 4: Fallback to synthetic noise
#     # =============================================
#     if (mode == 'synthetic') or (mode == 'hybrid' and saved_count < num_clips):
#         needed_clips = num_clips - saved_count
        
#         # Extract noise profile from quietest 10% of non-call audio
#         non_call_audio = y[exclusion_mask]
#         if len(non_call_audio) > sr:  # Need at least 1s
#             # Find quietest section
#             energy = librosa.feature.rms(y=non_call_audio, frame_length=2048, hop_length=512)[0]
#             quiet_frame = np.argmin(energy)
#             noise_profile = non_call_audio[quiet_frame*512 : (quiet_frame+4)*512]  # 4 frames ~93ms
            
#             # Generate noise matching this profile
#             for i in range(needed_clips):
#                 noise_clip = np.random.normal(scale=np.std(noise_profile), 
#                                             size=int(desired_clip_duration*sr))
                
#                 # Apply spectral shaping to match profile
#                 D_noise = librosa.stft(noise_clip)
#                 D_profile = librosa.stft(noise_profile)
#                 magnitudes = np.abs(D_profile).mean(axis=1)
#                 angles = np.angle(D_noise)
#                 reconstructed = librosa.istft(magnitudes[:, np.newaxis] * np.exp(1j * angles))
                
#                 output_path = os.path.join(output_folder, f"{base_name}_bg_synth_{saved_count+i}.wav")
#                 sf.write(output_path, reconstructed, sr)
        
#         saved_count += needed_clips

#     return saved_count

def process_audio_file(args):
    """
    Process a single audio file to extract bird calls and background segments.
    Modified to work with parallel processing.
    """
    audio_path, calls_output_folder, background_output_folder, params, verbose = args
    
    if not os.path.exists(audio_path):
        if verbose:
            print(f"File not found, skipping: {audio_path}")
        return 0
    
    if verbose:
        print(f"\nProcessing {audio_path}...")
    
    # Extract bird call segments
    call_intervals, y, sr, duration = extract_call_segments(
        audio_path, calls_output_folder,
        clip_duration=params['clip_duration'],
        sr=params['sr'],
        lowcut=params['lowcut'],
        highcut=params['highcut'],
        min_peak_distance=params['min_peak_distance'],
        height_percentile=params['height_percentile'],
        verbose=verbose
    )
    
    # Extract background segments
    extract_background_segments(
        audio_path, background_output_folder,
        call_intervals=call_intervals,
        desired_clip_duration=params['background_duration'],
        sr=params['sr'],
        call_extension=params['call_extension'],
        num_clips=params['num_background_clips'],
        verbose=verbose
    )
    
    if verbose:
        print(f"Completed processing {audio_path}")
    return len(call_intervals)

def find_audio_files(directory):
    """Find all audio files in the directory."""
    audio_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        for file in Path(directory).glob(f'**/*{ext}'):
            if file.exists() and file.is_file():
                audio_files.append(file)
    
    return sorted(audio_files)

def verify_background_clips(background_folder, output_folder=None, 
                          lowcut=2000, highcut=10000, sr=22050,
                          energy_threshold=2.0, bird_energy_ratio_threshold=0.3,
                          verbose=True):
    """
    Post-processing function to verify and clean background clips.
    Can either filter out clips with bird calls or create a new cleaned folder.
    
    Args:
        background_folder: Folder containing background clips
        output_folder: If provided, clean clips will be saved here. If None, bad clips are deleted.
        lowcut: Lower frequency bound for bird call detection
        highcut: Upper frequency bound for bird call detection
        sr: Sample rate
        energy_threshold: Threshold for energy variation (lower = stricter)
        bird_energy_ratio_threshold: Maximum ratio of bird-band energy to total (lower = stricter)
        verbose: Whether to print progress
        
    Returns:
        tuple: (total_clips, clean_clips) counts
    """
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        clean_mode = "copy"
    else:
        clean_mode = "delete"
    
    # Find all audio files in the background folder
    audio_files = []
    for ext in ['.wav', '.mp3', '.ogg', '.flac']:
        audio_files.extend(Path(background_folder).glob(f'**/*{ext}'))
    
    if not audio_files:
        if verbose:
            print(f"No audio files found in {background_folder}")
        return 0, 0
    
    if verbose:
        print(f"Verifying {len(audio_files)} background clips...")
    
    total_clips = len(audio_files)
    clean_clips = 0
    
    # Process each clip
    file_iter = tqdm(audio_files) if verbose else audio_files
    for audio_file in file_iter:
        try:
            # Load audio
            y, sr_file = librosa.load(str(audio_file), sr=sr)
            
            # Skip very short clips
            if len(y) < 0.5 * sr:
                if verbose:
                    print(f"Skipping {audio_file.name} - too short")
                continue
            
            # Apply multiple detection methods
            
            # 1. Check energy stability
            rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
            median_rms = np.median(rms)
            max_rms = np.max(rms)
            rms_ratio = max_rms / (median_rms + 1e-10)
            
            # 2. Check bird band energy
            y_bird = apply_bandpass_filter(y, lowcut, highcut, sr_file)
            bird_energy = np.sqrt(np.mean(y_bird ** 2))
            full_energy = np.sqrt(np.mean(y ** 2))
            bird_ratio = bird_energy / (full_energy + 1e-10)
            
            # 3. Check for peaks in the spectrum
            has_spectral_peaks = False
            try:
                # Compute spectrogram
                n_fft = 2048  # Higher resolution for verification
                S = np.abs(librosa.stft(y, n_fft=n_fft))
                
                # Calculate mean spectrum
                mean_spectrum = np.mean(S, axis=1)
                
                # Smooth spectrum to find peaks
                from scipy.signal import savgol_filter
                smooth_spectrum = savgol_filter(mean_spectrum, 11, 3)
                
                # Find peaks in bird frequency range
                freqs = librosa.fft_frequencies(sr=sr_file, n_fft=n_fft)
                bird_indices = np.where((freqs >= lowcut) & (freqs <= highcut))[0]
                
                if len(bird_indices) > 0:
                    bird_spectrum = smooth_spectrum[bird_indices]
                    # Find peaks
                    peaks, _ = find_peaks(bird_spectrum, prominence=np.std(bird_spectrum))
                    # If there are prominent peaks, it might be a bird call
                    has_spectral_peaks = len(peaks) > 0 and np.max(bird_spectrum) > 1.5 * np.median(bird_spectrum)
            except Exception as e:
                has_spectral_peaks = False
                if verbose:
                    print(f"Error in spectral analysis: {e}")
            
            # Determine if clip is clean
            is_clean = True
            
            # Reject if energy variation is too high (indicates transients/calls)
            if rms_ratio > energy_threshold:
                is_clean = False
                
            # Reject if too much energy in bird frequency band
            if bird_ratio > bird_energy_ratio_threshold:
                is_clean = False
                
            # Reject if clear spectral peaks in bird range
            if has_spectral_peaks:
                is_clean = False
            
            # Handle the clip based on cleanliness
            if is_clean:
                clean_clips += 1
                if clean_mode == "copy" and output_folder:
                    # Copy to clean folder
                    out_path = os.path.join(output_folder, audio_file.name)
                    try:
                        # If the format needs to change, use soundfile to save
                        if audio_file.suffix.lower() != '.wav':
                            sf.write(out_path, y, sr_file)
                        else:
                            # Otherwise just copy the file directly
                            shutil.copy2(str(audio_file), out_path)
                    except Exception as e:
                        if verbose:
                            print(f"Error copying {audio_file.name}: {e}")
            else:
                # Delete bad clips if in delete mode
                if clean_mode == "delete" and output_folder is None:
                    try:
                        os.remove(str(audio_file))
                    except Exception as e:
                        if verbose:
                            print(f"Error deleting {audio_file.name}: {e}")
                            
        except Exception as e:
            if verbose:
                print(f"Error processing {audio_file.name}: {e}")
    
    if verbose:
        if clean_mode == "copy":
            print(f"Verification complete: {clean_clips}/{total_clips} clips are clean")
            print(f"Clean clips saved to {output_folder}")
        else:
            print(f"Verification complete: {clean_clips}/{total_clips} clips are clean")
            print(f"Removed {total_clips - clean_clips} clips with bird calls")
    
    return total_clips, clean_clips

def process_bird_species(input_folder, output_base_folder, params, num_workers=None, verbose=True):
    """
    Process all audio files in a bird species folder using parallel processing.
    """
    # Get the species name from the folder name
    species_name = os.path.basename(input_folder)
    
    # Create species-specific output folder for calls
    calls_output_folder = os.path.join(output_base_folder, "calls", species_name)
    os.makedirs(calls_output_folder, exist_ok=True)
    
    # Background segments go to a shared backgrounds folder
    background_output_folder = os.path.join(output_base_folder, "backgrounds")
    os.makedirs(background_output_folder, exist_ok=True)
    
    # Find all audio files in the input folder
    audio_files = find_audio_files(input_folder)
    
    if not audio_files:
        if verbose:
            print(f"No audio files found in {input_folder}")
        return 0
    
    if verbose:
        print(f"Found {len(audio_files)} audio files in {input_folder}")
    
    # Prepare arguments for parallel processing
    process_args = []
    for audio_file in audio_files:
        process_args.append((
            str(audio_file),
            calls_output_folder,
            background_output_folder,
            params,
            verbose
        ))
    
    # Use parallel processing to process audio files
    if num_workers is None:
        # Use half of available cores by default
        num_workers = max(2, multiprocessing.cpu_count() // 2)
    
    total_calls = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Process files in parallel
        if verbose:
            results = list(tqdm(executor.map(process_audio_file, process_args), 
                           total=len(process_args), 
                           desc=f"Processing {species_name} files"))
        else:
            results = list(executor.map(process_audio_file, process_args))
        
        total_calls = sum(results)
    
    if verbose:
        print(f"\nProcessed {len(audio_files)} files for {species_name}, extracted {total_calls} calls")
    
    return total_calls

def batch_process(input_base_folder, output_base_folder, params, num_workers=None, verbose=True):
    """
    Process all bird species folders in the input base folder with parallel processing.
    """
    # Get all immediate subdirectories (bird species folders)
    species_folders = [f.path for f in os.scandir(input_base_folder) if f.is_dir()]
    
    if not species_folders:
        if verbose:
            print(f"No species folders found in {input_base_folder}")
            print("Checking if the input folder itself contains audio files...")
        
        # Try processing the input folder directly if it contains audio files
        audio_files = find_audio_files(input_base_folder)
        if audio_files:
            # Use the input folder name as the "species"
            species_name = os.path.basename(input_base_folder)
            calls_output_folder = os.path.join(output_base_folder, "calls", species_name)
            background_output_folder = os.path.join(output_base_folder, "backgrounds")
            
            os.makedirs(calls_output_folder, exist_ok=True)
            os.makedirs(background_output_folder, exist_ok=True)
            
            # Prepare arguments for parallel processing
            process_args = []
            for audio_file in audio_files:
                process_args.append((
                    str(audio_file),
                    calls_output_folder,
                    background_output_folder,
                    params,
                    verbose
                ))
            
            # Use parallel processing
            if num_workers is None:
                num_workers = max(2, multiprocessing.cpu_count() // 2)
            
            total_calls = 0
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                if verbose:
                    results = list(tqdm(executor.map(process_audio_file, process_args), 
                                      total=len(process_args),
                                      desc=f"Processing files"))
                else:
                    results = list(executor.map(process_audio_file, process_args))
                
                total_calls = sum(results)
            
            if verbose:
                print(f"Processed {len(audio_files)} files directly from {input_base_folder}, extracted {total_calls} calls")
            return
        
        if verbose:
            print("No audio files found.")
        return
    
    if verbose:
        print(f"Found {len(species_folders)} species folders to process")
    
    # Create the output base folder structure
    os.makedirs(os.path.join(output_base_folder, "calls"), exist_ok=True)
    os.makedirs(os.path.join(output_base_folder, "backgrounds"), exist_ok=True)
    
    # Process species folders in parallel
    total_species = 0
    total_calls = 0
    
    # Use ThreadPoolExecutor for folder-level parallelism
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(species_folders))) as executor:
        futures = []
        for species_folder in species_folders:
            future = executor.submit(
                process_bird_species, 
                species_folder, 
                output_base_folder, 
                params,
                num_workers,
                verbose
            )
            futures.append(future)
        
        # Collect results
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(species_folders),
                          desc="Processing species folders") if verbose else futures:
            species_calls = future.result()
            if species_calls > 0:
                total_species += 1
                total_calls += species_calls
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Batch processing complete!")
        print(f"Processed {total_species} species folders")
        print(f"Extracted {total_calls} total bird calls")
        print(f"All outputs saved to {output_base_folder}")

def main():
    """
    Main function to parse command-line arguments and run the batch processing.
    """
    parser = argparse.ArgumentParser(description="Bird Call Detector - Process audio files in a folder structure")
    
    # Input/output paths
    parser.add_argument("--input", "-i", required=True, help="Input base folder containing species subfolders")
    parser.add_argument("--output", "-o", required=True, help="Output base folder for extracted segments")
    
    # Audio processing parameters
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate (Hz)")
    parser.add_argument("--lowcut", type=int, default=500, help="Low cutoff frequency for bandpass filter (Hz)")
    parser.add_argument("--highcut", type=int, default=10000, help="High cutoff frequency for bandpass filter (Hz)")
    
    # Call detection parameters
    parser.add_argument("--clip-duration", type=float, default=3.0, help="Duration of extracted call clips (seconds)")
    parser.add_argument("--min-peak-distance", type=float, default=1.5, help="Minimum distance between detected calls (seconds)")
    parser.add_argument("--height-percentile", type=int, default=75, help="Only keep peaks above this percentile (0-100)")
    
    # Background extraction parameters
    parser.add_argument("--background-duration", type=float, default=3.0, help="Duration of background clips (seconds)")
    parser.add_argument("--min-segment-length", type=float, default=0.2, help="Minimum length of background segments (seconds)")
    parser.add_argument("--call-extension", type=float, default=0.5, help="Extension around call intervals to avoid bleed-over (seconds)")
    parser.add_argument("--num-background-clips", type=int, default=5, help="Number of background clips to extract per file")
    
    # Post-processing parameters
    parser.add_argument("--verify-backgrounds", action="store_true", help="Run post-processing verification on background clips")
    parser.add_argument("--energy-threshold", type=float, default=2.0, help="Maximum energy variation in background clips")
    parser.add_argument("--bird-energy-ratio", type=float, default=0.3, help="Maximum ratio of bird-band energy to total energy")
    parser.add_argument("--clean-output", help="Optional separate output folder for clean background clips")
    
    # Performance parameters
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: half of CPU cores)")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity of output")
    
    # Mode selection
    parser.add_argument("--only-verify", action="store_true", help="Only perform verification on existing background clips")
    
    args = parser.parse_args()
    
    # Set verbosity
    verbose = not args.quiet
    
    # Organize parameters into a dictionary
    params = {
        'sr': args.sr,
        'lowcut': args.lowcut,
        'highcut': args.highcut,
        'clip_duration': args.clip_duration,
        'min_peak_distance': args.min_peak_distance,
        'height_percentile': args.height_percentile,
        'background_duration': args.background_duration,
        'min_segment_length': args.min_segment_length,
        'call_extension': args.call_extension,
        'num_background_clips': args.num_background_clips
    }
    
    # Print parameter summary
    if verbose:
        print("\nBird Call Detector - Batch Processing")
        print(f"{'='*80}")
        print(f"Input folder: {args.input}")
        print(f"Output folder: {args.output}")
        print(f"Frequency range: {args.lowcut}-{args.highcut} Hz")
        print(f"Call clip duration: {args.clip_duration}s")
        print(f"Background clip duration: {args.background_duration}s")
        print(f"Number of workers: {args.workers if args.workers else 'auto'}")
        
        if args.verify_backgrounds or args.only_verify:
            print("\nBackground Verification Settings:")
            print(f"Energy threshold: {args.energy_threshold}")
            print(f"Bird energy ratio threshold: {args.bird_energy_ratio}")
            if args.clean_output:
                print(f"Clean backgrounds output: {args.clean_output}")
            
        print(f"{'='*80}\n")
    
    # Check for verify-only mode
    if args.only_verify:
        if verbose:
            print("Running in verification-only mode")
        
        background_folder = os.path.join(args.input, "backgrounds")
        if not os.path.isdir(background_folder):
            background_folder = args.input
            
        verify_background_clips(
            background_folder=background_folder,
            output_folder=args.clean_output,
            lowcut=args.lowcut,
            highcut=args.highcut,
            sr=args.sr,
            energy_threshold=args.energy_threshold,
            bird_energy_ratio_threshold=args.bird_energy_ratio,
            verbose=verbose
        )
        return
    
    # Run batch processing
    batch_process(args.input, args.output, params, args.workers, verbose)
    
    # Run verification if requested
    if args.verify_backgrounds:
        if verbose:
            print("\nPerforming post-processing verification on background clips")
        
        background_folder = os.path.join(args.output, "backgrounds")
        verify_background_clips(
            background_folder=background_folder,
            output_folder=args.clean_output,
            lowcut=args.lowcut,
            highcut=args.highcut,
            sr=args.sr,
            energy_threshold=args.energy_threshold,
            bird_energy_ratio_threshold=args.bird_energy_ratio,
            verbose=verbose
        )

if __name__ == "__main__":
    # Set a method to start subprocesses properly on Windows
    multiprocessing.set_start_method('spawn', force=True)
    main()