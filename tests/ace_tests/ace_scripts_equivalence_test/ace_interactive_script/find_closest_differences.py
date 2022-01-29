import util
from ase.io import read, write

ats = read("/home/eg475/scripts/tests/files/tiny_gap.train_set.xyz", ':')

distances = util.distances_dict(ats)

import pdb; pdb.set_trace()

for key, vals in distances.items():
    if key=='CC':
        continue
    print(f'{key}: {min(vals)}')
