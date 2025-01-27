import pandas as pd 
from pprint import pprint 
import requests 
import urllib.request
import os 
from tqdm import tqdm
import sys 

base_path = "/home/ste/Code/birds/dataset/"
base_url = "https://xeno-canto.org/api/2/recordings"

def download_species(scientific_name):
    genus, species = scientific_name.split(" ")
    dir_path = base_path+f"{genus}_{species}"
    os.makedirs(dir_path, exist_ok=True)

    page=1
    num_pages = sys.maxsize

    while(page<=num_pages):
        response = requests.get(base_url+"?query="+genus+"+"+species+"&page="+str(page)).json()
        num_pages = response["numPages"]
        
        print(scientific_name, page, "/", num_pages)
        recordings = response["recordings"]
        for recording in tqdm(recordings, total=len(recordings)):
            urllib.request.urlretrieve(recording["file"], dir_path+"/"+recording["file-name"])

        page = response["page"]+1
        


def download_dataset():

    df = pd.read_csv("breedingbirds.csv")
    scientific_names = df["Scientific Names"].to_list()

    for name in scientific_names:
        download_species(name)

    print("Finished!")

    
def get_number_of_records():

    df = pd.read_csv("breedingbirds.csv")
    scientific_names = df["Scientific Names"].to_list()

    xenocanto_data = {"Species":[], "numRecordings":[]}

    for name in tqdm(scientific_names, total=len(scientific_names)):
        bird = name.split(" ")
        response = requests.get(base_url+"?query="+bird[0]+"+"+bird[1]).json()
        # print(bird, response["numRecordings"], response["numSpecies"])
        xenocanto_data["Species"].append(bird)
        xenocanto_data["numRecordings"].append(response["numRecordings"])

    data = pd.DataFrame.from_dict(xenocanto_data)
    data.to_csv("numRecordings.csv")

if __name__=="__main__":
    download_dataset()