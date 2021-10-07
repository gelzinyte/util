import time
import os
from pathlib import Path
import re
import util

def track_mem(my_job_id, period=10, max_time=200000,
              out_fname='mem_usage.txt', cluster='womble'):

    start = time.time()
    no_checks = int(max_time / period)
    if os.path.isfile(out_fname):
        print('overwriting output file')
        os.remove(out_fname)

    with open(out_fname, 'w') as f:
        f.write(f'job_id: {my_job_id}\n')
        f.write('    no, time s, node no,   memory\n')

    for idx in range(no_checks):

        time.sleep(period)

        node_id = get_node_id(my_job_id, cluster=cluster)

        mem = None
        if node_id is not None:
            # job is running
            mem = get_mem_usage(node_id)

        if mem is None:
            mem = 'N/A'

        node_name = 'waiting'
        if node_id is not None:
            if cluster == 'womble':
                node_name = f'node{node_id}'
            else:
                node_name = node_id

        now = time.time()
        elapsed = int(now - start)
        with open(out_fname, 'a') as f:
            f.write(f'{idx:>6}, {elapsed:>6}, {node_name}, {mem:>8}\n')


def get_node_id(my_job_id, cluster='womble'):
    stdout, stderr = util.shell_stdouterr('qstat')
    job_id_pat = re.compile(r'^\d+')

    for line in stdout.splitlines():
        line = line.strip()
        job_id = job_id_pat.search(line)

        if job_id == None:
            continue

        elif job_id.group() == my_job_id:
            return node_id_from_line(line, cluster)

    print('Job has finished running, exiting!')
    exit(0)


def node_id_from_line(line, cluster='womble'):
    if cluster == 'womble':
        node_pat = re.compile(r'@node\d+')
        node_id_pat = re.compile(r"\d+$")
    elif cluster == 'young':
        node_pat = re.compile(r'node-[cyz]12[a-z]-\d{3}')
        node_id_pat = re.compile(r'node-[cyz]12[a-z]-\d{3}')



    node_match = node_pat.search(line)

    if node_match == None:
        return None

    else:
        return node_id_pat.search(node_match.group()).group()


def get_mem_usage(node_id):
    stdout, stderr = util.shell_stdouterr('qhost')

    for line in stdout.splitlines():

        elements = line.split()
        if elements[0] == node_id:
            return elements[8]


