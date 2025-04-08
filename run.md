```bash
# Option 1: Extract calls and backgrounds with improved algorithm
python script.py --input /path/to/bird_recordings --output /path/to/output_folder

# Option 2: Add post-processing verification during extraction
python script.py --input /path/to/bird_recordings --output /path/to/output_folder --verify-backgrounds

# Option 3: Only verify existing background clips (useful if you've already run extraction)
python script.py --input /path/to/background_clips --output /path/to/verified_output --only-verify

# Option 4: Extract to one folder but save verified clips to another
python script.py --input /path/to/bird_recordings --output /path/to/output_folder --verify-backgrounds --clean-output /path/to/clean_backgrounds
```