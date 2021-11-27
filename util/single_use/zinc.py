import subprocess
import os
import shutil
import re
import tempfile
from pathlib import Path
from tqdm import tqdm

import pandas as pd

import logging

logger = logging.getLogger(__name__)


def main(wget_fname, output_label, wdir='wdir'):

    log_fname = output_label + '.log'
    smi_output_fname = output_label + '.csv'
    wget_stdout = output_label + '.out'
    wget_stderr = output_label + '.err'


    Path(wdir).mkdir(parents=True, exist_ok=True)
    Path(log_fname).touch()

    with open(wget_fname, 'r') as f: commands = f.read().splitlines()
    with open(log_fname, 'r') as f: logged_labels = f.read().splitlines()

    staged_labels = []
    staged_info = []

    logger.info(f"wget file: {wget_fname}")
    for command in commands:
        label = get_label(command)
        if is_logged(label, logged_labels, staged_labels):
            continue
        staged_info.append(collect_info(command, label,
                             wget_stdout=wget_stdout, wget_stderr=wget_stderr,
                                        wdir=wdir))
        staged_labels.append(label)

    if len(staged_info) != 0 :
        staged_info_df = pd.concat(staged_info, ignore_index=True)
        staged_info_df.to_csv(smi_output_fname, mode='a', sep=' ')

        with open(log_fname, 'a') as f:
            f.write('\n'.join(staged_labels)+ '\n')
    else:
        logger.info("no new entries found")

def only_has_CH(entry):
    return not bool(re.search(r'[a-bd-gi-zA-BD-GI-Z]', entry))

def filter_for_CH(df):
    return df[df['smiles'].apply(only_has_CH)]

def collect_info(command, label, wget_stdout, wget_stderr, wdir):

    tmp_fname = tempfile.mkstemp(dir=wdir, suffix='.smi')[1]
    command = command.replace(f"-O {label}.smi", f"-O {tmp_fname}")

    logger.info(command)
    result = subprocess.run(command, shell=True, capture_output=True,
                            text=True)

    with open(wget_stdout, 'a') as f:
        f.write(result.stdout)
    with open(wget_stderr, 'a') as f:
        f.write(result.stderr)
    
    data = pd.read_csv(tmp_fname, delim_whitespace=True)

    data = filter_for_CH(data)

    os.remove(tmp_fname)
    return data


def is_logged(label, logged_labels, staged_labels):
    if label in logged_labels + staged_labels:
        return True
    return False


def get_label(command):
    pat = re.compile(r"wget http://files.docking.org/2D/[A-Z]{2}/[A-Z]{4}"
                     r".smi -O ([A-Z]{4}).smi")
    return pat.search(command).groups()[0]