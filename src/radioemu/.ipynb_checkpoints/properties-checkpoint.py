"""A module definining the static properties of the Emulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np



class EmulatorProperties:
    """A class that contains the properties of the emulator."""

    def __init__(self):
        here = Path(__file__).parent
        all_emulator_numbers = np.load('/home/dbreitman/Radio_Background/Models/Final_model/Feb_wPScsts.npz')
        self._data = all_emulator_numbers
        
        self.logPS_mean = all_emulator_numbers['logPS_mean']
        self.logPS_std = all_emulator_numbers['logPS_std']
        self.PS_ks = all_emulator_numbers['PS_k']
        self.PS_zs = all_emulator_numbers['PS_z']
        self.zs = all_emulator_numbers['redshifts']
        self.limits = np.array([[-2, 6], [33, 45], [-5, 0], [-6, -1], [0, 10]])
        self.logTb_std = all_emulator_numbers['Tb_std']
        self.logTb_mean = all_emulator_numbers["Tb_mean"]
        self.Tb_scale = all_emulator_numbers["Tb_scale"]
        self.logTr_mean = all_emulator_numbers["logTr_mean"]
        self.logTr_std = all_emulator_numbers["logTr_std"]
        
        self.mean_errors = np.load('/home/dbreitman/Radio_Background/Models/Final_model/median_test_errors.npz')



    @property
    def normalized_quantities(self) -> list[str]:
        """Return a list of the normalized quantities predicted by the emulator."""
        return [k.split("_")[0] for k in self._data if k.endswith("_mean")]


emulator_properties = EmulatorProperties()
