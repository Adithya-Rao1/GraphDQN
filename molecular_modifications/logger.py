import os
import logging

def setup_molecule_logger(log_dir='./logs', log_filename='molecular_mods.log'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  

    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)  

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger