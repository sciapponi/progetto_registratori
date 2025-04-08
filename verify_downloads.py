import pandas as pd
import os

def verify_downloads(csv_path, audio_dir):
    df = pd.read_csv(csv_path)
    downloaded_files = os.listdir(audio_dir)
    downloaded_ids = {f.split("_")[0] for f in downloaded_files}
    
    # Check which entries were downloaded
    df["downloaded"] = df["YTID"].apply(lambda x: x in downloaded_ids)
    
    print(f"Downloaded {df['downloaded'].sum()}/{len(df)} clips.")
    print("\nMissing clips:")
    missing = df[~df["downloaded"]]
    print(missing[["YTID", "start_seconds", "end_seconds"]].head())
    
    # Save verification report
    df.to_csv("download_verification.csv", index=False)
    print("\nSaved verification report to 'download_verification.csv'.")

# Usage
verify_downloads("rural_without_birds.csv", "downloaded_audio")