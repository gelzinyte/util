import subprocess
import os
import shutil
import re
import tempfile
from pathlib import Path
from tqdm import tqdm
import csv

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

    with open(smi_output_fname, 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(['smiles', 'zinc_id'])

    logger.info(f"wget file: {wget_fname}")
    for idx, command in enumerate(commands):
        logger.info(f'{command}, {idx}')
        label = get_label(command)
        if is_logged(label, logged_labels, staged_labels):
            continue
        staged_info.append(collect_info(command, label,
                             wget_stdout=wget_stdout, wget_stderr=wget_stderr,
                                        wdir=wdir))
        staged_labels.append(label)

        if idx%10 == 9:
            write_entries(staged_info, smi_output_fname, staged_labels,
                          log_fname)
            staged_info, staged_labels, logged_labels = reset_staged(
                log_fname)


    write_entries(staged_info, smi_output_fname, staged_labels,
                          log_fname)

def reset_staged(log_fname):
    with open(log_fname, 'r') as f:
        logged_labels = f.read().splitlines()
    return [], [], logged_labels


def write_entries(staged_info, smi_output_fname, staged_labels, log_fname):

    staged_info = [i for i in staged_info if i is not None]
    if len(staged_info) != 0 :
        staged_info_df = pd.concat(staged_info, ignore_index=True)
        staged_info_df.to_csv(smi_output_fname, mode='a', sep=' ',
                              index=False, header=False)

        logger.info(f"saved {len(staged_info)} entries")
    else:
        logger.info("no new entries found")

    with open(log_fname, 'a') as f:
        f.write('\n'.join(staged_labels) + '\n')


def only_has_CH(entry):
    return not bool(re.search(r'[a-bd-gi-zA-BD-GI-Z]', entry))

def only_has_CHO(entry):
    return not bool(re.search(r'[a-bd-gi-np-zA-BD-GI-NP-Z]', entry))


def filter_elements(df, elements):
    if elements=="CH":
        return df[df['smiles'].apply(only_has_CH)]
    elif elements=="CHO":
        return df[df['smiles'].apply(only_has_CHO)]

def collect_info(command, label, wget_stdout, wget_stderr, wdir):

    tmp_fname = tempfile.mkstemp(dir=wdir, suffix='.txt')[1]
    command = command.replace(f"-O {label}.txt", f"-O {tmp_fname}")

    logger.info(command)
    result = subprocess.run(command, shell=True, capture_output=True,
                            text=True)

    # import pdb; pdb.set_trace()
    with open(wget_stdout, 'a') as f:
        f.write(result.stdout)
    with open(wget_stderr, 'a') as f:
        f.write(result.stderr)

    filesize = Path(tmp_fname).stat().st_size
    if filesize > 0:
        data = pd.read_csv(tmp_fname, delim_whitespace=True)
        data = data[["smiles", "zinc_id"]]
        data = filter_elements(data, elements="CHO")

        if len(data) == 0:
            data = None

        os.remove(tmp_fname)
        return data
    else:
        logger.info(f"Found no entries in {command}")
        return None


def is_logged(label, logged_labels, staged_labels):
    if label in logged_labels + staged_labels:
        return True
    return False


def get_label(command):
    pat = re.compile(r"wget http://files.docking.org/2D/[A-Z]{2}/[A-Z]{4}"
                     r".txt -O ([A-Z]{4}).txt")
    return pat.search(command).groups()[0]