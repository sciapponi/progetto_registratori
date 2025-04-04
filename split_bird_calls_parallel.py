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
    
    return call_intervals, y, sr, duration

def extract_background_segments(audio_path, output_folder, call_intervals,
                           desired_clip_duration=3.0, sr=22050,
                           call_extension=1.0, num_clips=5,
                           energy_threshold=3.0, min_flatness=0.5,
                           verbose=True):
    """
    Optimized background extraction
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
    
    # Create extended call mask
    background_mask = np.ones(len(y), dtype=bool)
    try:
        for start, end in call_intervals:
            start_sample = max(0, int((start - call_extension) * sr))
            end_sample = min(len(y), int((end + call_extension) * sr))
            background_mask[start_sample:end_sample] = False
    except Exception as e:
        if verbose:
            print(f"Error processing call intervals: {e}")
        background_mask[:] = True  # Fallback to full audio

    # Optimize STFT computation - use lower resolution for faster processing
    n_fft = 1024  # Reduced from 1024
    hop_length = 512  # Increased from 256
    
    try:
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        flatness = (flatness - np.min(flatness)) / (np.max(flatness) - np.min(flatness) + 1e-10)
    except Exception as e:
        if verbose:
            print(f"Spectral analysis failed: {e}")
        flatness = np.ones(len(y) // hop_length + 1) * 0.5

    # Time-aligned flatness with interpolation
    try:
        frame_times = librosa.frames_to_time(np.arange(len(flatness)), 
                                           sr=sr, hop_length=hop_length)
        flatness_full = np.interp(np.linspace(0, duration, len(y)),
                                frame_times, flatness)
    except:
        flatness_full = np.ones(len(y)) * 0.5

    # Apply mask and find candidates
    flatness_masked = flatness_full * background_mask
    flatness_masked[flatness_masked < min_flatness] = 0  # Minimum quality threshold

    # Faster candidate selection - using strided operations
    stride = max(1, window_size // 10)  # Take 10% of window as stride for faster processing
    conv_size = len(flatness_masked) - window_size + 1
    
    if conv_size <= 0:
        if verbose:
            print("Audio too short for background extraction")
        return 0
        
    # Sample at regular intervals rather than full convolution
    sample_points = np.arange(0, conv_size, stride)
    if len(sample_points) == 0:
        sample_points = [0]
        
    # Calculate mean values at sample points
    sample_values = np.array([np.mean(flatness_masked[i:i+window_size]) for i in sample_points])
    candidates = sample_points[np.argsort(-sample_values)[:num_clips*2]]

    # Filter candidates with energy analysis
    clean_clips = []
    candidate_desc = tqdm(candidates, desc="Processing candidates") if verbose else candidates
    
    for pos in candidate_desc:
        try:
            if pos + window_size > len(y):
                continue

            clip = y[pos:pos+window_size]
            
            # Efficient energy analysis - use fewer frames
            energy = librosa.feature.rms(y=clip, frame_length=1024, hop_length=512)[0]
            if len(energy) < 3:
                continue
                
            median_energy = np.median(energy)
            if np.max(energy) > median_energy * energy_threshold:
                continue
                
            # Apply fade in/out
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

def process_audio_file(args):
    """
    Process a single audio file to extract bird calls and background segments.
    Modified to work with parallel processing.
    """
    audio_path, calls_output_folder, background_output_folder, params, verbose = args
    
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
        audio_files.extend(Path(directory).glob(f'**/*{ext}'))
    
    return sorted(audio_files)

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
    
    # Performance parameters
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: half of CPU cores)")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity of output")
    
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
    
    # Set verbosity
    verbose = not args.quiet
    
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
        print(f"{'='*80}\n")
    
    # Run batch processing
    batch_process(args.input, args.output, params, args.workers, verbose)

if __name__ == "__main__":
    # Set a method to start subprocesses properly on Windows
    multiprocessing.set_start_method('spawn', force=True)
    main()