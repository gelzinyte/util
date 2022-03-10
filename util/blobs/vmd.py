import os
from pathlib import Path
import time	
import shutil 
import subprocess
import click

def main(density_dir):

	root_dir = Path(__file__).parent

	template_fnames = [root_dir / "scripts/plot_template_isosurfaces.vmd",
					   root_dir / "scripts/plot_template_density_cross_section.vmd"]

	plot_dirs = ["vmd_plots_isosurface", "vmd_plots_crossection"]

	density_dir = Path(density_dir).resolve()

	homedir = os.getcwd()

	for vmd_template_fname, plot_dir in zip(template_fnames, plot_dirs):
	
		if not os.path.isdir(plot_dir):
			os.makedirs(plot_dir)
		os.chdir(plot_dir)

		density_fnames = [os.path.join(density_dir, fname) for fname in
						  os.listdir(density_dir) if 'cube' in fname]

		for fname in density_fnames:

			stem = os.path.splitext(os.path.basename(fname))[0]
			
			target_fname = stem + '.tga'
			
			if os.path.exists(target_fname):
				continue
			else:
				print(f'---making {target_fname}')

			if 'eldens' in fname or 'electron_density' in fname:
				isosurface = 0.35
			elif 'spindens' in fname or "spin_density" in fname:
				isosurface = 0.001

			with open(vmd_template_fname, 'r') as f:
				template = f.read()

			template = template.replace('<cube_filename>', fname)
			template = template.replace('<isosurface_val>', str(isosurface))
			label = os.path.basename(os.path.splitext(fname)[0])
			template = template.replace('<label>', label)

			with open('tmp.vmd', 'w') as f:
				f.write(template)

			exec = '/Applications/VMD\ ' \
				   '1.9.4a51-x86_64-Rev9.app/Contents/vmd/vmd_MACOSXX86_64 -e ' \
				   'tmp.vmd '

			subprocess.run(exec, shell=True)

			os.remove(stem + '.dat')

			# time.sleep(10)

		os.chdir(homedir)



if __name__ == '__main__':
	main()



