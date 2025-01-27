import requests
from tqdm import tqdm
# Function to get audio observations
def get_audio_observations(taxon_id, per_page=100):
    base_url = "https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_id": taxon_id,  # Taxon ID for the species
        "media_type": "sound",  # Filter for sound recordings
        "per_page": per_page,  # Results per page
        "page": 1,  # Start at page 1
    }
    
    audio_files = []
    while True:
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Error: Unable to fetch data (HTTP {response.status_code})")
            break
        
        data = response.json()
        results = data.get("results", [])
        
        # Extract audio file links from results
        for observation in tqdm(results, total = len(results)):
            sounds = observation.get("sounds", [])
            for sound in sounds:
                audio_files.append({
                    "file": sound["file_url"],  # Audio file URL
                    "file-name": sound.get("file_name", f"observation_{observation['id']}.mp3"),
                })
        
        # Check if there's another page
        if not data["total_results"] or len(results) < per_page:
            break
        params["page"] += 1  # Move to the next page
    
    return audio_files

# Example usage
if __name__ == "__main__":
    taxon_id = 18301  # Replace with the taxon ID for your species
    audio_files = get_audio_observations(taxon_id)

    if audio_files:
        print(f"Found {len(audio_files)} audio recordings:")
        for audio in audio_files:
            print(f"- {audio['file']} (filename: {audio['file-name']})")
    else:
        print("No audio recordings found.")
