"""Module that interacts with the Emulator PyTorch model."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import nn
from .inputs import EmulatorInput
from .outputs import EmulatorOutput
from .outputs import RawEmulatorOutput
from .properties import emulator_properties

from .ml import Radio_Emulator
log = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
    
class Emulator:
    r"""A class that loads an emulator and uses it to obtain 21cmFAST summaries.

    Parameters
    ----------
    version : str, optional
        Emulator version to use/download, default is 'latest'.
    """

    def __init__(self, version: str = "latest"):
        
        model = Radio_Emulator()
        model.load_state_dict(torch.load('/home/dbreitman/Radio_Background/Models/Final_model/FINAL_EMULATOR', map_location=device))
        model.eval()

        self.model = model
        self.inputs = EmulatorInput()
        self.properties = emulator_properties

    def __getattr__(self, name: str) -> Any:
        """Allow access to emulator properties directly from the emulator object."""
        return getattr(self.properties, name)

    def predict(
        self, astro_params: ParamVecType, verbose: bool = False
    ) -> tuple[np.ndarray, EmulatorOutput, dict[str, np.ndarray]]:
        r"""Call the emulator, evaluate it at the given parameters, restore dimensions.

        Parameters
        ----------
        astro_params : np.ndarray or dict
            An array with the nine astro_params input all $\in [0,1]$ OR in the
            21cmFAST AstroParams input units. Dicts (e.g. p21.AstroParams.defining_dict)
            are also accepted formats. Arrays of only dicts are accepted as well
            (for batch evaluation).
        verbose : bool, optional
            If True, prints the emulator prediction.

        Returns
        -------
        theta : np.ndarray
            The normalized parameters used to evaluate the emulator.
        emu : EmulatorOutput
            The emulator output, with dimensions restored.
        errors : dict
            The mean error on the test set (i.e. independent of theta).
        """
        theta = self.inputs.make_param_array(astro_params, normed=True)
        emu = RawEmulatorOutput(self.model(torch.Tensor(theta).to(device)).detach().cpu().numpy())
        emu = emu.get_renormalized()

        return theta, emu
