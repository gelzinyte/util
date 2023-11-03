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


def main(wget_fname, output_label, elements, wdir='wdir'):

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

    if not Path(smi_output_fname).exists():
        with open(smi_output_fname, 'w') as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerow(['smiles', 'zinc_id'])

    logger.info(f"wget file: {wget_fname}")
    for idx, command in enumerate(commands):
        logger.info(f'{command}, {idx}')
        # if idx == 28:
            # import pdb; pdb.set_trace()
        label = get_label(command)
        if is_logged(label, logged_labels, staged_labels):
            continue
        staged_info.append(collect_info(command, label, wget_stdout=wget_stdout, 
                                        wget_stderr=wget_stderr,wdir=wdir, elements=elements))
        staged_labels.append(label)

        if idx%10== 9:
            write_entries(staged_info, smi_output_fname, staged_labels,log_fname)
            staged_info, staged_labels, logged_labels = reset_staged(log_fname)


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
    if isinstance(entry, int):
        entry = str(entry)
    return not bool(re.search(r'[a-bd-gi-np-zA-BD-GI-NP-Z]', entry))


def only_has_CHNOPS(entry):
    # import pdb; pdb.set_trace()
    # print(entry)
    return not bool(re.search(r'[a-bd-gi-mq-rt-zA-BD-GI-MQ-RT-Z]', entry))

def filter_elements(df, elements):
    if elements=="CH":
        return df[df['smiles'].apply(only_has_CH)]
    elif elements=="CHO":
        return df[df['smiles'].apply(only_has_CHO)]
    elif elements=="CHNOPS":
        return df[df["smiles"].apply(only_has_CHNOPS)]

def collect_info(command, label, wget_stdout, wget_stderr, wdir, elements):

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
    upper_limit = 200 * 1024**2 # 500 MB
    # if filesize > 0 and filesize < upper_limit:
    #     # data = pd.read_csv(tmp_fname, delim_whitespace=True, engine='python') # python bc pd bug
    #     data = pd.read_csv(tmp_fname, delim_whitespace=True, usecols=["smiles", "zinc_id"])
    #     data = data[["smiles", "zinc_id"]]
    #     data = filter_elements(data, elements=elements)

    #     if len(data) == 0:
    #         data = None

    #     os.remove(tmp_fname)
    #     return data

    # elif filesize > upper_limit:
    if filesize > 0:
        data = read_line_by_line(tmp_fname, elements)
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


def read_line_by_line(tmp_fname, elements):

    out_data = []

    with open(tmp_fname, 'r') as f:
        for idx, line in enumerate(f):

            if idx == 0:
                entries = line.split() 
                assert entries[0] == "smiles"
                assert entries[1] == "zinc_id"
                continue

            entries = line.split()
            smiles_str = entries[0]
            zinc_id = entries[1]

            if elements == "CH":
                if only_has_CH(smiles_str):
                    out_data.append({"smiles":smiles_str, "zinc_id":zinc_id})
            elif elements == "CHO":
                if only_has_CHO(smiles_str):
                    # import pdb; pdb.set_trace()
                    out_data.append({"smiles":smiles_str, "zinc_id":zinc_id})
            else:
                raise ValueError(f"elements {elements} not supported")
    
    if len(out_data) == 0:
        return None
    else:
        df = pd.DataFrame(out_data)
        return df









