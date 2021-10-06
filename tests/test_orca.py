from util.calculators import orca
import os
import numpy as np

def ref_path():
    return os.path.abspath(os.path.dirname(__file__))



def test_read_xyz_from_output():

    input = os.path.join(ref_path(), 'files/orca.out')
    at = orca.read_xyz_from_output(input)

    expected_symbols = ['O', 'C', 'C', 'C', 'N', 'C', 'N', 'H', 'H', 'H', 'H',
                        'C', 'C', 'N', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'N',
                        'Fe', 'N', 'C', 'C', 'C', 'N', 'C', 'C', 'C', 'C',
                        'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
                        'H', 'H', 'H', 'H', 'H']

    expected_positions = np.array([[ 8.369100e-02,  4.320900e-01, 5.353961e+00],
                                     [ 5.114400e-02,  3.507790e-01,  4.213516e+00],
                                     [ 8.037550e-01,  4.262110e-01, -2.928732e+00],
                                     [ 4.047710e-01,  1.251300e-01, -4.208254e+00],
                                     [-7.822430e-01, -5.713520e-01, -4.071004e+00],
                                     [-1.061758e+00, -6.710560e-01, -2.745794e+00],
                                     [-1.187180e-01, -7.638000e-02, -2.038759e+00],
                                     [ 1.683856e+00,  9.653890e-01, -2.587558e+00],
                                     [ 8.459550e-01,  3.387710e-01, -5.178828e+00],
                                     [-1.347611e+00, -9.457190e-01, -4.823859e+00],
                                     [-1.938917e+00, -1.170688e+00, -2.339296e+00],
                                     [-1.645361e+00, -3.926735e+00,  2.696920e-01],
                                     [-1.754007e+00, -2.482331e+00,  2.144280e-01],
                                     [-5.101300e-01, -1.920476e+00,  8.986000e-02],
                                     [ 3.847460e-01, -2.957805e+00,  6.690200e-02],
                                     [-3.160200e-01, -4.222382e+00,  1.713230e-01],
                                     [ 1.768564e+00, -2.831332e+00, -2.116400e-02],
                                     [ 2.470603e+00, -1.630241e+00, -7.329300e-02],
                                     [ 3.916219e+00, -1.529426e+00, -1.133450e-01],
                                     [ 4.215675e+00, -1.976810e-01, -1.265650e-01],
                                     [ 2.951791e+00,  5.117630e-01, -1.051440e-01],
                                     [ 1.911295e+00, -3.798040e-01, -7.396300e-02],
                                     [-6.637500e-02,  5.631600e-02,  1.034000e-03],
                                     [-2.049253e+00,  4.851450e-01, -2.728800e-02],
                                     [-2.611796e+00,  1.726901e+00, -1.640210e-01],
                                     [-1.911636e+00,  2.921461e+00, -3.110530e-01],
                                     [-5.256780e-01,  3.053223e+00, -3.128210e-01],
                                     [ 3.722740e-01,  2.025238e+00, -1.959760e-01],
                                     [ 1.618724e+00,  2.593830e+00, -2.135930e-01],
                                     [ 2.824076e+00,  1.898092e+00, -1.538210e-01],
                                     [ 1.508626e+00,  4.033034e+00, -3.481580e-01],
                                     [ 1.755240e-01,  4.317915e+00, -4.171810e-01],
                                     [-4.056815e+00,  1.627292e+00, -1.119470e-01],
                                     [-4.352500e+00,  3.066710e-01,  6.966700e-02],
                                     [-3.087180e+00, -3.987860e-01,  1.165590e-01],
                                     [-2.957284e+00, -1.780289e+00,  2.442120e-01],
                                     [-2.489148e+00, -4.610386e+00,  3.688770e-01],
                                     [ 1.646990e-01, -5.201492e+00,  1.752250e-01],
                                     [ 2.354746e+00, -3.753949e+00, -2.651000e-02],
                                     [ 4.598327e+00, -2.380083e+00, -1.241220e-01],
                                     [ 5.196481e+00,  2.787410e-01, -1.540430e-01],
                                     [-2.500239e+00,  3.837125e+00, -4.112200e-01],
                                     [ 3.745060e+00,  2.486270e+00, -1.751990e-01],
                                     [ 2.354486e+00,  4.719966e+00, -3.896400e-01],
                                     [-3.074630e-01,  5.289909e+00, -5.248670e-01],
                                     [-4.741282e+00,  2.471857e+00, -1.977870e-01],
                                     [-5.331751e+00, -1.646360e-01,  1.629490e-01],
                                     [-3.875940e+00, -2.362437e+00,  3.530070e-01]])


    assert np.all(list(at.symbols) == expected_symbols)
    assert np.all(list(at.positions) == expected_positions)



