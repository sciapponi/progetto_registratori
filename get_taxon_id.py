import requests
import pandas as pd
from tqdm import tqdm
# List of species names
species_list = ["Aegolius funereus", "Strix nebulosa", "Tyto alba"]
df = pd.read_csv("breedingbirds.csv")
species_list = df["Scientific Names"].to_list()
# Base URL for iNaturalist taxa API
base_url = "https://api.inaturalist.org/v1/taxa"

# Dictionary to store results
taxon_ids = {}

for species in tqdm(species_list, total=len(species_list)):
    # API query
    response = requests.get(base_url, params={"q": species})
    if response.status_code == 200:
        data = response.json()
        if data["results"]:
            # Get the first taxon result
            taxon = data["results"][0]
            taxon_ids[species] = taxon["id"]
        else:
            taxon_ids[species] = None
    else:
        taxon_ids[species] = None

df["taxon_id"] = taxon_ids.values()
# df.drop("numRecordings")
df.to_csv("taxon.csv")
# Print taxon IDs
for species, taxon_id in taxon_ids.items():
    print(f"{species}: {taxon_id}")
