import shutil
import glob
from pathlib import Path
import filecmp


def archive_file(fnames_ll, cluster):

    if cluster == "csd3":
        dest_dir_root = "/home/eg475/backups/"
        source_beginning = "/rds/user/eg475/hpc-work/"
    else:
        raise AttributeError(f"cluster {cluster} is not yet supported")

    # multiple can be given to click as list
    for fnames in fnames_ll:
        # each can be a glob
        for source_fname in glob.iglob(fnames, recursive=True):

            source_fname = Path(source_fname).resolve()
            
            if source_fname.is_dir():
                print(f"is dir, skipping: {source_fname}")
                continue

            assert source_beginning in str(source_fname)

            dest_fname = Path(str(source_fname).replace(source_beginning, dest_dir_root))
            dest_dir = dest_fname.parent
            dest_dir.mkdir(exist_ok=True, parents=True)

            if dest_fname.exists():
                # compare two files
                is_the_same = filecmp.cmp(source_fname, dest_fname)
                if is_the_same:
                    print(f"already backed up: {dest_fname}")
                    continue
                else:
                    raise RuntimeError(f"found dest file {dest_fname} that is not the same as the rouce file {source_fname}, aborting")


            print(f"copying \n > {source_fname}\n > {dest_fname}")
            shutil.copy2(source_fname, dest_fname)

            


