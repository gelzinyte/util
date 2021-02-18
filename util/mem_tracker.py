import time
import os
from pathlib import Path
import re
import util

def track_mem(my_job_id, period=10, max_time=100000, out_fname='mem_usage.txt'):

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

        node_no = get_node_no(my_job_id)

        mem = None
        if node_no is not None:
            # job is running
            mem = get_mem_usage(node_no)

        if mem is None:
            mem = 'N/A'

        node_name = 'waiting'
        if node_no is not None:
            node_name = f'node{node_no}'

        now = time.time()
        elapsed = int(now - start)
        with open(out_fname, 'a') as f:
            f.write(f'{idx:>6}, {elapsed:>6}, {node_name}, {mem:>8}\n')


def get_node_no(my_job_id):
    stdout, stderr = util.shell_stdouterr('qstat')
    job_id_pat = re.compile(r'^\d+')

    for line in stdout.splitlines():
        line = line.strip()
        job_id = job_id_pat.search(line)

        if job_id == None:
            continue

        elif job_id.group() == my_job_id:
            return node_no_from_line(line)

    print('Job has finished running, exiting!')
    exit(0)


def node_no_from_line(line):
    node_pat = re.compile(r'@node\d+')
    node_no_pat = re.compile(r"\d+$")

    node_match = node_pat.search(line)

    if node_match == None:
        return None

    else:
        return node_no_pat.search(node_match.group()).group()


def get_mem_usage(node_no):
    stdout, stderr = util.shell_stdouterr('qhost')

    for line in stdout.splitlines():

        elements = line.split()
        if elements[0] == f'node{node_no}':
            return elements[8]


