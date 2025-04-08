import csv
import os
import yt_dlp
import logging
from yt_dlp.utils import download_range_func # Helper for segment times
import logging # Ensure logging is imported if not already
# --- Configuration ---
CSV_FILE = 'rural_without_birds.csv'
OUTPUT_DIR = 'downloaded_audio'
AUDIO_FORMAT = 'mp3'  # Desired audio format (e.g., 'mp3', 'm4a', 'opus', 'wav')
AUDIO_QUALITY = '192' # Desired audio quality in kbit/s (for lossy formats)
# Set to True to skip download if the file already exists
SKIP_EXISTING = True
# Set to True to stop on the first download error, False to continue
STOP_ON_ERROR = False

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Function ---
def format_time(seconds_float):
    """Converts seconds (float) to HH:MM:SS.fff format for FFmpeg"""
    try:
        seconds_float = float(seconds_float)
        if seconds_float < 0:
            return "00:00:00.000" # Handle potential negative start times
        milliseconds = int((seconds_float - int(seconds_float)) * 1000)
        seconds_int = int(seconds_float)
        minutes, seconds = divmod(seconds_int, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    except ValueError:
        logging.error(f"Invalid time value encountered: {seconds_float}")
        return None # Indicate error

# --- Main Download Logic ---
def download_audio_segments(csv_path, output_dir):
    """
    Reads the CSV file and downloads the specified audio segments.
    """
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {os.path.abspath(output_dir)}")

    downloaded_count = 0
    skipped_count = 0
    error_count = 0

    try:
        with open(csv_path, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader) # Skip the header row

            # Try to find column indices robustly
            try:
                ytid_idx = header.index('YTID')
                start_idx = header.index('start_seconds')
                end_idx = header.index('end_seconds')
            except ValueError as e:
                logging.error(f"Missing required column in CSV header: {e}")
                logging.error(f"Expected columns: 'YTID', 'start_seconds', 'end_seconds'. Found: {header}")
                return

            for i, row in enumerate(reader):
                # Basic validation: Ensure row has enough columns
                if len(row) <= max(ytid_idx, start_idx, end_idx):
                    logging.warning(f"Skipping row {i+2}: Not enough columns. Row content: {row}")
                    continue

                ytid = row[ytid_idx].strip()
                start_sec_str = row[start_idx].strip()
                end_sec_str = row[end_idx].strip()

                # Validate YTID format (basic check)
                if not ytid or len(ytid) != 11: # Standard YTIDs are 11 chars
                     # AudioSet YTIDs might look different, relax this check if needed
                     # Let's just check if it's not empty for now.
                     if not ytid:
                         logging.warning(f"Skipping row {i+2}: Invalid or empty YTID '{ytid}'. Row: {row}")
                         continue


                # Validate and convert times
                try:
                    start_sec = float(start_sec_str)
                    end_sec = float(end_sec_str)
                    if start_sec < 0 or end_sec < 0 or end_sec <= start_sec:
                        raise ValueError("Invalid time range")
                except ValueError:
                    logging.warning(f"Skipping row {i+2}: Invalid start/end seconds ('{start_sec_str}', '{end_sec_str}'). Row: {row}")
                    continue

                # Construct YouTube URL and output filename
                video_url = f'https://www.youtube.com/watch?v={ytid}'
                # Use integer seconds for cleaner filenames if they are whole numbers
                start_f = int(start_sec) if start_sec.is_integer() else start_sec
                end_f = int(end_sec) if end_sec.is_integer() else end_sec
                output_filename_template = os.path.join(output_dir, f'{ytid}_{start_f}-{end_f}.%(ext)s')

                final_output_path = output_filename_template.replace('%(ext)s', AUDIO_FORMAT)

                # Skip if file already exists (optional)
                if SKIP_EXISTING and os.path.exists(final_output_path):
                    logging.info(f"Skipping already downloaded file: {os.path.basename(final_output_path)}")
                    skipped_count += 1
                    continue

                logging.info(f"Attempting download: {ytid} ({start_sec:.2f}s - {end_sec:.2f}s)")

                # --- yt-dlp options ---
                # Note: Using download_ranges with FFmpeg arguments is generally robust for segments.
                # yt-dlp will instruct FFmpeg to seek accurately.

                ffmpeg_args = [
                     '-ss', str(start_sec), # Start time
                     '-to', str(end_sec)    # End time
                    ]
                # Alternative using HH:MM:SS.fff format - sometimes more reliable with FFmpeg versions
                # start_time_fmt = format_time(start_sec)
                # end_time_fmt = format_time(end_sec)
                # if start_time_fmt and end_time_fmt:
                #     ffmpeg_args = ['-ss', start_time_fmt, '-to', end_time_fmt]
                # else:
                #     logging.error(f"Could not format time for row {i+2}, skipping. Row: {row}")
                #     error_count += 1
                #     if STOP_ON_ERROR: break
                #     continue


                ydl_opts = {
                    'format': f'bestaudio[ext={AUDIO_FORMAT}]/bestaudio/best',
                    'outtmpl': output_filename_template,
                    'quiet': True,
                    'no_warnings': True,
                    'download_ranges': download_range_func(None, [(start_sec, end_sec)]),
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': AUDIO_FORMAT,
                        'preferredquality': AUDIO_QUALITY,
                    }, {
                        'key': 'FFmpegMetadata',
                        'add_metadata': True,
                    }, {
                        'key': 'EmbedThumbnail',
                        'already_have_thumbnail': False,
                    }],
                    'ffmpeg_location': None,
                    'keepvideo': False,
                    'noplaylist': True,
                    'logtostderr': False,
                }



                # --- Execute Download ---
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([video_url])
                    logging.info(f"Successfully downloaded and processed: {os.path.basename(final_output_path)}")
                    downloaded_count += 1
                except yt_dlp.utils.DownloadError as e:
                    logging.error(f"Error downloading {ytid} ({start_sec:.2f}-{end_sec:.2f}s): {e}")
                    error_count += 1
                    # Clean up potentially incomplete file if error occurred
                    if os.path.exists(final_output_path):
                       try:
                           os.remove(final_output_path)
                           logging.info(f"Removed potentially incomplete file: {os.path.basename(final_output_path)}")
                       except OSError as remove_err:
                           logging.warning(f"Could not remove incomplete file {os.path.basename(final_output_path)}: {remove_err}")
                    # Check if partial download file exists (often ends with .part)
                    part_file = final_output_path + ".part"
                    if os.path.exists(part_file):
                        try:
                           os.remove(part_file)
                           logging.info(f"Removed partial download file: {os.path.basename(part_file)}")
                        except OSError as remove_err:
                           logging.warning(f"Could not remove partial file {os.path.basename(part_file)}: {remove_err}")

                    if STOP_ON_ERROR:
                        logging.error("Stopping script due to error.")
                        break
                except Exception as e: # Catch other potential errors
                    logging.error(f"An unexpected error occurred for row {i+2} ({ytid}): {e}")
                    error_count += 1
                    if STOP_ON_ERROR:
                        logging.error("Stopping script due to unexpected error.")
                        break

    except FileNotFoundError:
        logging.error(f"Could not open CSV file: {csv_path}")
        return
    except StopIteration:
        logging.error(f"CSV file '{csv_path}' seems empty or only contains a header.")
        return
    except Exception as e:
        logging.error(f"An critical error occurred during CSV processing: {e}")
        return

    logging.info("--------------------")
    logging.info("Download process finished.")
    logging.info(f"Successfully downloaded: {downloaded_count}")
    logging.info(f"Skipped (already exist): {skipped_count}")
    logging.info(f"Errors: {error_count}")
    logging.info("--------------------")


# --- Run the script ---
if __name__ == "__main__":
    download_audio_segments(CSV_FILE, OUTPUT_DIR)