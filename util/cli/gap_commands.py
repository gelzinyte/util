import click
import numpy as np

@click.command('mem')
@click.option('--num-desc', '-nx', type=click.INT, help='number of descriptors', multiple=True)
@click.option('--num-der', '-ndx', type=click.INT, help='number of derrivatives of descriptors', multiple=True)
@click.option('--lmax', '-l', type=click.INT, help='l_max')
@click.option('--nmax', '-n', type=click.INT, help='n_max')
@click.option('--n-elements', '-el', type=click.INT, help='number of  elements')
@click.option('--n-prop', '-p', type=click.INT, multiple=True, help="number of target properties")
@click.option('--n-sparse', '-sp', type=click.INT, multiple=True, )
def estimate_mem(num_desc, num_der, lmax, nmax, n_elements, n_prop, n_sparse):

    GB = 1024**3

    # descriptors and gradients
    num_desc = np.sum(num_desc)
    num_der = np.sum(num_der)
    desc_dim = (lmax + 1) * (nmax * n_elements + 1) * (nmax * n_elements) / 2 
    mem_desc = (num_desc +  num_der) * desc_dim * 8 / GB # Gbytes

    # covariance matrices
    # mem_cov = np.sum(n_prop) * n_sparse * 8 / GB

    print(f'estimated memory: {mem_desc:.1f} GB')
