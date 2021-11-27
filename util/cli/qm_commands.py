import click

@click.command('read-orca')
@click.option('--output-xyz', '-o', help='output filename')
@click.option('--orca-label', '-l', help='prefix to orca files to read from')
@click.option('--input-xyz', '-i', help='xyz')
def read_orca_stuff(output_xyz, orca_label, input_xyz):
    from util import qm
    at = qm.read_orca_output(orca_label, input_xyz)
    write(output_xyz, at)

@click.command('cu-cy')
@click.option('--structures-dir', show_default=True,
              help='where are the input xyz to be calculated on',
              default='/data/eg475/carbyne/optimised_structures' )
@click.option('--uks-orca-template-fname', show_default=True,
              default='/data/eg475/carbyne/calculate/uks_cc_template.inp' )
@click.option('--cc-orca-template-fname', show_default=True,
              default='/data/eg475/carbyne/calculate/dlpno_ccsd_template.inp' )
@click.option('--sub-template-fname', show_default=True,
              default='/data/eg475/carbyne/calculate/sub_template.sh' )
@click.option('--output-dir', show_default=True,
              default='/data/eg475/carbyne/calculate/outputs' )
@click.option('--task' )#, type=click.Choice(['calculate, density_plots']) )
@click.option('--submit', is_flag=True)
def calculate(structures_dir,
        uks_orca_template_fname,
         cc_orca_template_fname,
         sub_template_fname,
         output_dir,
            task,
         submit=False):

    from util import cc_results

    cc_results.main(structures_dir=structures_dir,
                         uks_orca_template_fname=uks_orca_template_fname,
                         cc_orca_template_fname=cc_orca_template_fname,
                         sub_template_fname=sub_template_fname,
                         output_dir=output_dir,
                         task=task,
                         submit=submit)



@click.command('scf-conv')
@click.argument('orca-output')
@click.option('--method', '-m', default='dft', help='method for iterations')
@click.option('--plot-fname', '-o', default='orca_scf_convergence.png')
def plot_scf_convergence_graph(orca_output, method, plot_fname):
    qm.orca_scf_plot(orca_output, method, plot_fname)
