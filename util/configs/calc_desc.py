import yaml
import wfl.calc_descriptor
from wfl.configset import ConfigSet, ConfigSet_out


def from_param_yaml(input, gap_fit_param_yaml):
    with open(gap_fit_param_yaml) as f:
        gap_fit_params = yaml.safe_load(f)
    descriptors = gap_fit_params.pop('_gap')
    inputs = ConfigSet(input_configs=input)
    outputs = ConfigSet_out()
    wfl.calc_descriptor.calc(inputs=inputs, outputs=outputs,
                             descs=descriptors, key='soap',
                        local=True, verbose=True)

    print(outputs.output_configs[0][0].arrays.keys())





