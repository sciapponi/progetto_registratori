import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import find_peaks, butter, filtfilt
import argparse
from pathlib import Path
import shutil
from tqdm import tqdm

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
    
    Returns a list of call intervals as (start_time, end_time), along with the audio data, sample rate, and duration.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load audio
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
        print(f"Saved call clip: {filename}")

        call_intervals.append((start_time, end_time))
    
    return call_intervals, y, sr, duration

def extract_background_segments(audio_path, output_folder, call_intervals,
                           desired_clip_duration=3.0, sr=22050,
                           call_extension=1.0, num_clips=5,
                           energy_threshold=3.0, min_flatness=0.5):
    """
    Robust background extraction with:
    - Bird call filtering
    - Graceful error handling
    - Progress feedback
    - Adaptive fallbacks
    """
    
    # Safe output folder creation
    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        print(f"Could not create output folder: {e}")
        return 0

    # Validate inputs
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return 0

    if desired_clip_duration <= 0:
        print("Clip duration must be positive")
        return 0

    # Load audio with error handling
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return 0

    if len(y) == 0:
        print("Empty audio file")
        return 0

    duration = len(y) / sr
    window_size = int(desired_clip_duration * sr)
    
    # Handle case where audio is shorter than desired clip
    if window_size > len(y):
        print(f"Audio too short ({duration:.1f}s) for {desired_clip_duration}s clips")
        window_size = len(y)
        desired_clip_duration = window_size / sr

    # Create extended call mask
    background_mask = np.ones(len(y), dtype=bool)
    try:
        for start, end in call_intervals:
            start_sample = max(0, int((start - call_extension) * sr))
            end_sample = min(len(y), int((end + call_extension) * sr))
            background_mask[start_sample:end_sample] = False
    except Exception as e:
        print(f"Error processing call intervals: {e}")
        background_mask[:] = True  # Fallback to full audio

    # Compute spectral features with fallbacks
    try:
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        flatness = (flatness - np.min(flatness)) / (np.max(flatness) - np.min(flatness) + 1e-10)
    except Exception as e:
        print(f"Spectral analysis failed: {e}")
        # Fallback to uniform flatness
        flatness = np.ones(len(y) // 256 + 1) * 0.5

    # Time-aligned flatness with interpolation
    try:
        frame_times = librosa.frames_to_time(np.arange(len(flatness)), 
                                           sr=sr, hop_length=256)
        flatness_full = np.interp(np.linspace(0, duration, len(y)),
                                frame_times, flatness)
    except:
        flatness_full = np.ones(len(y)) * 0.5

    # Apply mask and find candidates
    flatness_masked = flatness_full * background_mask
    flatness_masked[flatness_masked < min_flatness] = 0  # Minimum quality threshold

    # Find candidate segments
    try:
        conv = np.convolve(flatness_masked, np.ones(window_size), 'valid')
        candidates = np.argsort(conv)[-num_clips*3:]  # Extra candidates for filtering
    except:
        candidates = [0]  # Fallback to first segment

    # Filter candidates with energy analysis
    clean_clips = []
    for pos in tqdm(candidates, desc="Processing candidates"):
        try:
            if pos + window_size > len(y):
                continue

            clip = y[pos:pos+window_size]
            
            # Energy analysis
            energy = librosa.feature.rms(y=clip, frame_length=512, hop_length=128)[0]
            if len(energy) < 3:  # Too short for analysis
                continue
                
            median_energy = np.median(energy)
            if np.max(energy) > median_energy * energy_threshold:
                continue  # Skip clips with transients
                
            # Spectral quality check
            clip_flatness = np.mean(flatness_full[pos:pos+window_size])
            if clip_flatness < min_flatness:
                continue

            # Apply fade
            fade = min(512, window_size // 10)
            clip[:fade] *= np.linspace(0, 1, fade)
            clip[-fade:] *= np.linspace(1, 0, fade)
            
            clean_clips.append(clip)
            
            if len(clean_clips) >= num_clips:
                break
        except Exception as e:
            print(f"Error processing candidate: {e}")
            continue

    # Save valid clips
    saved_count = 0
    for i, clip in enumerate(clean_clips):
        try:
            output_path = os.path.join(output_folder, 
                                     f"{os.path.splitext(os.path.basename(audio_path))[0]}_bg_{i}.wav")
            sf.write(output_path, clip, sr)
            saved_count += 1
        except Exception as e:
            print(f"Error saving clip {i}: {e}")

    print(f"Successfully saved {saved_count}/{num_clips} background clips")
    return saved_count

def process_audio_file(audio_path, calls_output_folder, background_output_folder, params):
    """
    Process a single audio file to extract bird calls and background segments.
    """
    print(f"\n{'='*80}\nProcessing {audio_path}...")
    
    # Extract bird call segments
    call_intervals, y, sr, duration = extract_call_segments(
        audio_path, calls_output_folder,
        clip_duration=params['clip_duration'],
        sr=params['sr'],
        lowcut=params['lowcut'],
        highcut=params['highcut'],
        min_peak_distance=params['min_peak_distance'],
        height_percentile=params['height_percentile']
    )
    
    # Extract background segments
    # extract_background_segments_with_concatenation(
    #     audio_path, background_output_folder,
    #     call_intervals=call_intervals,
    #     desired_clip_duration=params['background_duration'],
    #     min_segment_length=params['min_segment_length'],
    #     sr=params['sr'],
    #     call_extension=params['call_extension'],
    #     lowcut=params['lowcut'],
    #     highcut=params['highcut'],
    #     num_background_clips=params['num_background_clips']
    # )
    extract_background_segments(
        audio_path, background_output_folder,
        call_intervals=call_intervals,
        desired_clip_duration=params['background_duration'],
        sr=params['sr'],
        call_extension=params['call_extension'],
        num_clips=params['num_background_clips']
    )
    
    print(f"Completed processing {audio_path}")
    return len(call_intervals)

def find_audio_files(directory):
    """Find all audio files in the directory."""
    audio_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(Path(directory).glob(f'**/*{ext}'))
    
    return sorted(audio_files)

def process_bird_species(input_folder, output_base_folder, params):
    """
    Process all audio files in a bird species folder.
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
        print(f"No audio files found in {input_folder}")
        return 0
    
    print(f"Found {len(audio_files)} audio files in {input_folder}")
    total_calls = 0
    
    # Process each audio file
    for audio_file in audio_files:
        num_calls = process_audio_file(
            str(audio_file),
            calls_output_folder,
            background_output_folder,
            params
        )
        total_calls += num_calls
    
    print(f"\nProcessed {len(audio_files)} files for {species_name}, extracted {total_calls} calls")
    return total_calls

def batch_process(input_base_folder, output_base_folder, params):
    """
    Process all bird species folders in the input base folder.
    """
    # Get all immediate subdirectories (bird species folders)
    species_folders = [f.path for f in os.scandir(input_base_folder) if f.is_dir()]
    
    if not species_folders:
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
            
            total_calls = 0
            for audio_file in audio_files:
                num_calls = process_audio_file(
                    str(audio_file),
                    calls_output_folder,
                    background_output_folder,
                    params
                )
                total_calls += num_calls
                
            print(f"Processed {len(audio_files)} files directly from {input_base_folder}, extracted {total_calls} calls")
            return
        
        print("No audio files found.")
        return
    
    print(f"Found {len(species_folders)} species folders to process")
    
    # Create the output base folder structure
    os.makedirs(os.path.join(output_base_folder, "calls"), exist_ok=True)
    os.makedirs(os.path.join(output_base_folder, "backgrounds"), exist_ok=True)
    
    # Process each species folder
    total_species = 0
    total_calls = 0
    
    for species_folder in species_folders:
        species_calls = process_bird_species(species_folder, output_base_folder, params)
        if species_calls > 0:
            total_species += 1
            total_calls += species_calls
    
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
    
    args = parser.parse_args()
    
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
    print("\nBird Call Detector - Batch Processing")
    print(f"{'='*80}")
    print(f"Input folder: {args.input}")
    print(f"Output folder: {args.output}")
    print(f"Frequency range: {args.lowcut}-{args.highcut} Hz")
    print(f"Call clip duration: {args.clip_duration}s")
    print(f"Background clip duration: {args.background_duration}s")
    print(f"{'='*80}\n")
    
    # Run batch processing
    batch_process(args.input, args.output, params)

if __name__ == "__main__":
    main()