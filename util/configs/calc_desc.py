import yaml
from wfl.configset import ConfigSet, OutputSpec
from wfl.descriptors import quippy


def from_param_yaml(input, gap_fit_param_yaml):
    with open(gap_fit_param_yaml) as f:
        gap_fit_params = yaml.safe_load(f)
    descriptors = gap_fit_params.pop('_gap')
    inputs = ConfigSet(input_configs=input)
    outputs = OutputSpec()
    quippy.calc(inputs=inputs, outputs=outputs,
                             descs=descriptors, key='soap',
                        local=True, verbose=True)

    print(outputs.output_configs[0][0].arrays.keys())





