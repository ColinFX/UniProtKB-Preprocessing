"""
Extract features from downloaded UniProtKB json files and merge them into a single jsonl file for each split.
Features not mentioned in the downloaded json files will be set to an empty string in the output jsonl file.

Sequences longer than 1024 amino-acid tokens will be segmented into overlapping sequences of length 1024 with 
overlapping length 256. Each record in the output jsonl file thus corresponds to a segmented sequence.
"""

import json
import os
import re
from typing import Any, Dict, List

from tqdm import tqdm


READ_JSON_ROOT_DIR = "/ssd1/UniProtKB/download"  # expected to contain /test, /val and /train
SAVE_JSON_DIR = "/ssd1/UniProtKB/processed"


def get_features(data: Dict[str, Any]) -> Dict[str, str]: 
    """Extract features from a UniProtKB json file then clean up the text."""
    gathered_dict: Dict[str, str] = {}
    
    for key in [
        "accession", "sequence", 
        "organism", "family", "domain", 
        "location", "subunit", "activity", "cofactor", "ptm", 
        "pathway", "tissue", "induction",
        "description"
    ]: 
        try: 
            function_name = f"_get_{key}"
            gathered_dict[key] = globals()[function_name](data)
            gathered_dict[key] = _remove_pubmed_annotation(gathered_dict[key])
        except KeyError: 
            gathered_dict[key] = ""
            print(f"KeyError: {key} encountered in protein {_get_accession(data)}")

    return gathered_dict


def _get_accession(data: Dict[str, Any]) -> str: 
    return data["primaryAccession"]


def _get_sequence(data: Dict[str, Any]) -> str:  
    return data["sequence"]["value"]


def _get_name(data: Dict[str, Any]) -> str: 
    return data["proteinDescription"]["recommendedName"]["fullName"]["value"]


def _get_description(data: Dict[str, Any]) -> str: 
    comments: List[Dict[str, Any]] = data["comments"]
    functions = [
        text["value"] 
        for comment in comments if comment["commentType"] == "FUNCTION"
        for text in comment["texts"]
    ]
    return " | ".join(functions)


def _get_organism(data: Dict[str, Any]) -> str: 
    return f"lineage: {data['organism']['lineage'][-1]}, organism: {data['organism']['scientificName']}"


def _get_family(data: Dict[str, Any]) -> str: 
    comments: List[Dict[str, Any]] = data["comments"]
    similarities = [
        text["value"]
        for comment in comments if comment["commentType"] == "SIMILARITY"
        for text in comment["texts"]
    ]
    return " | ".join(similarities)


def _get_domain(data: Dict[str, Any]) -> str: 
    comments: List[Dict[str, Any]] = data["comments"]
    domains = [
        text["value"]
        for comment in comments if comment["commentType"] == "DOMAIN"
        for text in comment["texts"]
    ]
    return " | ".join(domains)


def _get_location(data: Dict[str, Any]) -> str: 
    def _format_subcellular_location(subcellular_location_data: Dict[str, Any]) -> str:
        location = (
            subcellular_location_data["location"]["value"]
            if "location" in subcellular_location_data.keys()
            else "unknown"
        )
        topology = (
            subcellular_location_data["topology"]["value"]
            if "topology" in subcellular_location_data.keys()
            else "unknown"
        )
        orientation = (
            subcellular_location_data["orientation"]["value"]
            if "orientation" in subcellular_location_data.keys()
            else "unknown"
        )
        return f"location: {location}; topology: {topology}; orientation: {orientation}"

    comments: List[Dict[str, Any]] = data["comments"]
    locations = [
        _format_subcellular_location(subcellular_location)
        for comment in comments if (
            comment["commentType"] == "SUBCELLULAR LOCATION" and "subcellularLocations" in comment.keys()
        )
        for subcellular_location in comment["subcellularLocations"]
    ]
    notes = [
        text["value"]
        for comment in comments if (comment["commentType"] == "SUBCELLULAR LOCATION" and "note" in comment.keys())
        for text in comment["note"]["texts"]
    ]
    return " | ".join(locations + notes)


def _get_subunit(data: Dict[str, Any]) -> str: 
    comments: List[Dict[str, Any]] = data["comments"]
    subunits = [
        text["value"]
        for comment in comments if comment["commentType"] == "SUBUNIT"
        for text in comment["texts"]
    ]
    return " | ".join(subunits)


def _get_activity(data: Dict[str, Any]) -> str: 
    comments: List[Dict[str, Any]] = data["comments"]
    activities = [
        comment["reaction"]["name"] for comment in comments
        if comment["commentType"] == "CATALYTIC ACTIVITY"
    ]
    return " | ".join(activities)


def _get_cofactor(data: Dict[str, Any]) -> str: 
    comments: List[Dict[str, Any]] = data["comments"]
    cofactors = [
        cofactor["name"]
        for comment in comments if (comment["commentType"] == "COFACTOR" and "cofactors" in comment.keys())
        for cofactor in comment["cofactors"]
    ]
    notes = [
        text["value"]
        for comment in comments if (comment["commentType"] == "COFACTOR" and "note" in comment.keys())
        for text in comment["note"]["texts"]
    ]
    return " | ".join(cofactors + notes)


def _get_ptm(data: Dict[str, Any]) -> str: 
    comments: List[Dict[str, Any]] = data["comments"]
    ptms = [
        text["value"]
        for comment in comments if comment["commentType"] == "PTM"
        for text in comment["texts"]
    ]
    return " | ".join(ptms)


def _get_pathway(data: Dict[str, Any]) -> str: 
    comments: List[Dict[str, Any]] = data["comments"]
    pathways = [
        text["value"]
        for comment in comments if comment["commentType"] == "PATHWAY"
        for text in comment["texts"]
    ]
    return " | ".join(pathways)


def _get_tissue(data: Dict[str, Any]) -> str: 
    comments: List[Dict[str, Any]] = data["comments"]
    tissues = [
        text["value"]
        for comment in comments if comment["commentType"] == "TISSUE SPECIFICITY"
        for text in comment["texts"]
    ]
    return " | ".join(tissues)


def _get_induction(data: Dict[str, Any]) -> str: 
    comments: List[Dict[str, Any]] = data["comments"]
    interactions = [
        text["value"]
        for comment in comments if comment["commentType"] == "INDUCTION"
        for text in comment["texts"]
    ]
    return " | ".join(interactions)


def _remove_pubmed_annotation(text: str) -> str:
    """Remove innermost parentheses containing the word 'PubMed'."""
    pattern = r'\([^()]*PubMed[^()]*\)'
    return re.sub(pattern, '', text)


def segment_with_overlapping(sequence: str, max_len: int, overlap_len: int) -> List[str]: 
    """
    Segment a sequence into overlapping sequences of length max_len if the sequence is longer than max_len. 
    The last segment will always contain the last max_len tokens. 
    """
    if len(sequence) <= max_len:
        return [sequence]
    else: 
        segments = []
        for start_pos in range(0, len(sequence), max_len - overlap_len): 
            segment = sequence[start_pos : start_pos+max_len]
            if len(segment) == max_len:
                segments.append(segment)
        if len(sequence) > max_len:
            segments.append(sequence[-max_len:])
        return segments


if __name__ == "__main__":
    for split in ["test", "val", "train"]:
        read_json_dir = os.path.join(READ_JSON_ROOT_DIR, split)
        save_json_path = os.path.join(SAVE_JSON_DIR, f"{split}.jsonl")
        if not os.path.exists(SAVE_JSON_DIR):
            os.makedirs(SAVE_JSON_DIR)

        with open(save_json_path, "w") as save_file:

            for file_name in tqdm(os.listdir(read_json_dir), postfix=f"{split}"):
                try: 
                    with open(os.path.join(read_json_dir, file_name), "r") as read_file:
                        data = json.load(read_file)
                except Exception as e: 
                    print(f"{e} -- Failed to load {file_name}")
                    continue

                processed_data = get_features(data)
                
                segmented_sequences = segment_with_overlapping(
                    processed_data["sequence"], max_len=1022, overlap_len=256  # (1024 - 2) exclude start and end tokens
                )

                for segmented_sequence in segmented_sequences:
                    save_file.write(
                        json.dumps(
                            {
                                "accession": processed_data["accession"],
                                "sequence": segmented_sequence,
                                "organism": processed_data["organism"],
                                "family": processed_data["family"],
                                "domain": processed_data["domain"],
                                "location": processed_data["location"],
                                "subunit": processed_data["subunit"],
                                "activity": processed_data["activity"],
                                "cofactor": processed_data["cofactor"],
                                "ptm": processed_data["ptm"],
                                "pathway": processed_data["pathway"],
                                "tissue": processed_data["tissue"],
                                "induction": processed_data["induction"],
                                "description": processed_data["description"],
                            }
                        ) + "\n"
                    )
