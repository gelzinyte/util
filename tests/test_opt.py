from util import opt
from wfl.configset import ConfigSet, OutputSpec

# input input fixtures from conftest.py
def test_optimise(atoms, gap_calculator):

    atoms.set_distance(0, 1, 1.3)

    inputs = ConfigSet(input_configs=[atoms])
    outputs = OutputSpec()


    opt.optimise(inputs=inputs,
                 outputs=outputs,
                 traj_step_interval=1,
                 calculator=gap_calculator,
                 output_prefix='gappy_')

    # returned full trajectory
    assert len([at for at in outputs.to_ConfigSet()]) == 6

    outputs = OutputSpec()
    opt.optimise(inputs=inputs,
                 outputs=outputs,
                 traj_step_interval=None,
                 calculator=gap_calculator,
                 output_prefix='gappy_')


    # returned only last config
    assert len([at for at in outputs.to_ConfigSet()]) == 1
