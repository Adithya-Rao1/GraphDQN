"""Contains constants used throughout ADMET-AI."""
from importlib import resources


# Paths to data and models

DEFAULT_ADMET_PATH = "Meits_Data/admet_data/admet.csv"
DEFAULT_DRUGBANK_PATH = "Metis_Data/admet_data/drugbank_approved.csv" 
DEFAULT_MODELS_DIR = "Metis_Data/admet_models" 

# DrugBank columns
DRUGBANK_ID_COLUMN = "id"
DRUGBANK_NAME_COLUMN = "name"
DRUGBANK_SMILES_COLUMN = "smiles"
DRUGBANK_ATC_PREFIX = "atc"
DRUGBANK_ATC_NAME_PREFIX = "atc_name"
DRUGBANK_ATC_CODE_COLUMN = DRUGBANK_ATC_PREFIX
DRUGBANK_DELIMITER = ";"