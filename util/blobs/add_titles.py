import os
import subprocess
import click
from tqdm import tqdm

@click.command()
@click.option('--input-dir')
@click.option('--output-dir')
def main(input_dir, output_dir):

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    filenames = [os.path.join(input_dir, fname) for fname in os.listdir(
        input_dir)]

    for filename in tqdm(filenames):

        base = os.path.basename(os.path.splitext(filename)[0])
        target_fname = os.path.join(output_dir, base + '.pdf')

        exec = f"convert {filename} -gravity North -draw 'scale 4,4 text 0," \
               f"10 {base}' {target_fname}"

        subprocess.run(exec, shell=True)



if __name__=='__main__':
    main()
