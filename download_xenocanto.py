import pandas as pd
from pprint import pprint
import requests
import urllib.request
import os
from tqdm import tqdm
import sys
from time import sleep
import signal

base_path = "/media/ste/New Volume/dataset/"
base_url = "https://xeno-canto.org/api/2/recordings"
stop_requested = False


def signal_handler(signum, frame):
    """Handle Ctrl+C signal."""
    global stop_requested
    stop_requested = True
    print("\nGracefully shutting down...")


# Register signal handler for SIGINT
signal.signal(signal.SIGINT, signal_handler)


def fetch_json_with_retries(url, retries=3, backoff_factor=2):
    """Fetch JSON data from a URL with retries on failure."""
    for attempt in range(retries):
        if stop_requested:
            return None
        try:
            response = requests.get(url, timeout=10)  # Set a timeout for the request
            if response.status_code == 200:
                return response.json()
            else:
                print(f"HTTP error {response.status_code}: {response.reason}")
                break  # Don't retry for HTTP errors
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if attempt < retries - 1:
                sleep(backoff_factor ** attempt)  # Exponential backoff
            else:
                print(f"Failed to fetch {url} after {retries} retries")
    return None


def download_species(scientific_name):
    global stop_requested
    genus, species = scientific_name.split(" ")
    dir_path = base_path + f"{genus}_{species}"
    os.makedirs(dir_path, exist_ok=True)

    page = 1
    num_pages = sys.maxsize

    while page <= num_pages:
        if stop_requested:
            break
        url = f"{base_url}?query={genus}+{species}&page={page}"
        response = fetch_json_with_retries(url)

        if not response:
            print(f"Skipping {scientific_name} due to failed API response")
            return

        num_pages = response["numPages"]
        print(scientific_name, page, "/", num_pages)
        recordings = response["recordings"]

        for recording in tqdm(recordings, total=len(recordings)):
            if stop_requested:
                break
            try:
                file_url = recording["file"]
                file_path = os.path.join(dir_path, recording["file-name"])
                urllib.request.urlretrieve(file_url, file_path)
            except Exception as e:
                print(f"Failed to download file {file_url}: {e}")

        if stop_requested:
            break
        page = response["page"] + 1


def download_dataset():
    global stop_requested
    try:
        df = pd.read_csv("breedingbirds.csv")
    except FileNotFoundError:
        print("CSV file not found. Please ensure 'breedingbirds.csv' exists.")
        return

    scientific_names = df["Scientific Names"].to_list()

    try:
        for name in scientific_names:
            if stop_requested:
                print("Stopping download due to user request.")
                break
            download_species(name)
    finally:
        # Cleanup or save any intermediate state here if needed
        print("Finished or interrupted. Exiting cleanly.")


def get_number_of_records():
    global stop_requested
    try:
        df = pd.read_csv("breedingbirds.csv")
    except FileNotFoundError:
        print("CSV file not found. Please ensure 'breedingbirds.csv' exists.")
        return

    scientific_names = df["Scientific Names"].to_list()
    xenocanto_data = {"Species": [], "numRecordings": []}

    for name in tqdm(scientific_names, total=len(scientific_names)):
        if stop_requested:
            break
        bird = name.split(" ")
        url = f"{base_url}?query={bird[0]}+{bird[1]}"
        response = fetch_json_with_retries(url)

        if response:
            xenocanto_data["Species"].append(name)
            xenocanto_data["numRecordings"].append(response.get("numRecordings", 0))
        else:
            print(f"Failed to fetch data for {name}. Skipping...")

    if not stop_requested:
        data = pd.DataFrame.from_dict(xenocanto_data)
        data.to_csv("numRecordings.csv")


if __name__ == "__main__":
    try:
        download_dataset()
    except KeyboardInterrupt:
        print("\nDownload interrupted. Cleanup done.")
