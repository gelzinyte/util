import logging

import numpy as np

from ase import Atoms
from ase import units

from wfl.generate_configs.vib import Vibrations

logger = logging.getLogger(__name__)



def sample_downweighted_normal_modes(inputs, outputs, temp, sample_size, prop_prefix,
                        info_to_keep=None, arrays_to_keep=None):
    """Multiple times displace along normal modes for all atoms in input

    Parameters
    ----------

    inputs: Atoms / list(Atoms) / ConfigSet_in
        Structures with normal mode information (eigenvalues &
        eigenvectors)
    outputs: ConfigSet_out
    temp: float
        Temperature for normal mode displacements
    sample_size: int
        Now many perturbed structures per input structure to return
    prop_prefix: str / None
        prefix for normal_mode_frequencies and normal_mode_displacements
        stored in atoms.info/arrays
    info_to_keep: str, default "config_type"
        string of Atoms.info.keys() to keep
    arrays_to_keep: str, default None
        string of Atoms.arrays.keys() entries to keep

    Returns
    -------
    """

    if isinstance(inputs, Atoms):
        inputs = [inputs]

    for atoms in inputs:
        at_vib = Vibrations(atoms, prop_prefix)

        energies_into_modes = downweight_energies(at_vib.frequencies,
                                                                  temp)
        try:
            sample = at_vib.sample_normal_modes(energies_for_modes=energies_into_modes[6:],
                                                sample_size=sample_size,
                                                info_to_keep=info_to_keep,
                                                arrays_to_keep=arrays_to_keep)
        except TypeError as e:
            print(f'config type: {at_vib.atoms.info["config_type"]}')
            raise(e)
        except ValueError as e:
            logger.info(f'could not sample {at_vib.atoms.info["config_type"]}, adding original structure')
            sample = atoms
        outputs.write(sample)

    outputs.end_write()


def downweight_energies(frequencies_eV, temp, threshold_invcm=200,
                                      threshold_eV=None):

    assert threshold_invcm is None or threshold_eV is None

    if threshold_invcm is not None:
        threshold_eV = threshold_invcm * units.invcm

    weights = []
    for freq in frequencies_eV:
        if freq < threshold_eV and freq > 0:
            weights.append(weight_function(freq, midway=threshold_eV / 2))
        elif freq <= 0:
            weights.append(0)
        else:
            weights.append(1)
    weights = np.array(weights)

    orig_energies = units.kB * temp

    return weights * orig_energies


def weight_function(xs, midway=100):
    """weights for energies. Midway for where the inflection point should
    be. Goes to zero at zero. """
    xs_for_p = shifted_to_p_domain(xs, midway)
    y_values = p(p(p(xs_for_p)))
    shifted_ys = shift_y(y_values)
    return shifted_ys

def shifted_to_p_domain(xs, midway=100):
    """Goes from xs in the range of (0, 2*midway) to (-1, 1) to be
    applicable to the polynomial function."""
    return xs / midway - 1

def p(x):
    """polynomial function """
    return 3/2 * x - 1/2 * x ** 3

def shift_y(ys):
    """shift results from [-1, 1] to [0, 1]"""
    return (ys + 1 ) * 0.5



