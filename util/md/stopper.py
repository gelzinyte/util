from wfl.generate.md.abort import AbortSimBase
from util import configs

class BadGeometry(AbortSimBase):
    pass
    def __init__(self, n_failed_steps=5, mult=1.2, skin=0, mark_elements=False,
        info_label=None):
        super().__init__(n_failed_steps)

        self.mult = mult
        self.skin = skin
        self.mark_elements = mark_elements
        self.info_label = info_label


    def check_if_atoms_ok(self, at):
        geometry_ok =  configs.check_geometry(at, mult=self.mult, skin=self.skin,
        mark_elements=self.mark_elements)
        if self.info_label is not None:
            at.info[self.info_label] = geometry_ok
        return geometry_ok
