import os
import time	
import shutil 
import subprocess
import click

#@click.command()
#@click.option('--vmd_template_fname', show_default=True,
#			  default='/Users/elena/code/carbyne_comulene/plot_template.vmd')
#@click.option('--plot-dir', show_default=True,
#			  default='/Users/elena/code/carbyne_comulene/vmd_plots')
#@click.option('--density-dir', show_default=True,
#			  default='/Users/elena/code/carbyne_comulene/all_200x200x200_densities')
#def main(vmd_template_fname, plot_dir, density_dir):
def main():

	template_fnames = ["/Users/elena/code/carbyne_comulene/plot_template_isosurfaces.vmd",
						"/Users/elena/code/carbyne_comulene/plot_template_density_cross_section.vmd"]
	plot_dirs = ["vmd_plots_isosurface", "vmd_plots_crossection"]

	density_dir = '/Users/elena/code/carbyne_comulene/all_200x200x200_densities'


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

			if 'eldens' in fname:
				isosurface = 0.35
			elif 'spindens' in fname:
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
				   'tmp.vmd'

			subprocess.run(exec, shell=True)

			os.remove(stem + '.dat')

		os.chdir(homedir)



if __name__ == '__main__':
	main()



