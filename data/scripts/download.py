"""
Download json files from UniProtKB API using accessions from corresponding splits. 
Execute the python script from the root directory of the project.
"""

import json
import os
import requests

from tqdm import tqdm


READ_TXT_ROOT_DIR = "./data"  # expected to contain test.txt, val.txt and train.txt
SAVE_JSON_ROOT_DIR = "/ssd1/UniProtKB/download"  # expected to contain /test, /val and /train
BASE_URL = "https://rest.uniprot.org/uniprotkb"  # see: https://www.uniprot.org/help/uniprotkb


if __name__ == "__main__": 
    for split in ["test", "val", "train"]: 
        save_json_dir = os.path.join(SAVE_JSON_ROOT_DIR, split)
        if not os.path.exists(save_json_dir):
            os.makedirs(save_json_dir)

        read_txt_path = os.path.join(READ_TXT_ROOT_DIR, f"{split}.txt")
        with open(read_txt_path, "r") as file:
            accessions = [line.strip() for line in file]

        for accession in tqdm(accessions, postfix=f"{split}"):
            url = f'{BASE_URL}/{accession}.json'
            response = requests.get(url)

            # for status code, see: https://en.wikipedia.org/wiki/List_of_HTTP_status_codes
            if response.status_code == 200:  
                data = response.json()
                save_json_path = os.path.join(save_json_dir, f"{accession}.json")
                with open(save_json_path, "w") as f:
                    json.dump(data, f, indent=4)
            
            else:
                print(f"Failed to download {accession}")
                continue
