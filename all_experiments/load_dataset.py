from __future__ import absolute_import

import subprocess
import os
from tqdm import tqdm
from utils import setup_dqn_logger

def load_dataset(logger, wget_file, output_dir, num_smiles_per_file=70, log=False):
    def read_wget_file():
        """Reads the .wget file and returns all lines."""
        with open(wget_file, 'r') as f:
            return f.readlines()

    def download_smiles(lines):
        """Executes wget commands from the selected lines, saving all files in a single directory."""
        os.makedirs(output_dir, exist_ok=True)
        for line in lines:
            # Extract the wget URL and output file name
            parts = line.strip().split("wget")
            if len(parts) > 1:
                wget_command = parts[1].split("-O")
                if len(wget_command) > 1:
                    url = wget_command[0].strip()
                    file_name = wget_command[1].strip().split("/")[-1]
                    # Modify the wget command to save directly in output_dir
                    command = f"wget {url} -O {os.path.join(output_dir, file_name)}"
                    subprocess.run(command, shell=True)

    def collect_smiles_files(output_dir, num_smiles_per_file=None):
        """Collects SMILES strings from all downloaded .smi files and reads their contents."""
        smiles = []
        
        for root, _, files in tqdm(os.walk(output_dir), desc='Collecting SMILES:'):
            for file in files:
                if file.endswith('.smi'):
                    with open(os.path.join(root, file), 'r') as f:
                        lines = f.readlines()
                        for line in lines[1:]:  # Skip header
                            parts = line.strip().split()
                            smiles.append(parts[0])  # Append the SMILES string

                            # Check if we've reached the desired limit
                            if num_smiles_per_file and len(smiles) >= num_smiles_per_file*1440:
                                return smiles
    
    if os.path.exists(output_dir):
        if log:
            logger.info("Existing SMILES directory found.")
        smiles_list = collect_smiles_files(output_dir, num_smiles_per_file)
    else:
        if log:
            logger.info("Creating SMILES directory.")
        lines = read_wget_file()
        download_smiles(lines)
        smiles_list = collect_smiles_files(output_dir, num_smiles_per_file)
    
    return smiles_list