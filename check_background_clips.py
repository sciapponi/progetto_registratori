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