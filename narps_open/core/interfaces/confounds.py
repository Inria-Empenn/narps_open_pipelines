#!/usr/bin/python
# coding: utf-8

""" This module defines custom interfaces related to confounds computation """

from os.path import abspath

from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, traits, TraitedSpec, ImageFile, File
    )
from DVARS import DVARS_Calc

class ComputeDVARSInputSpec(BaseInterfaceInputSpec):
    """ Input specifications of a ComputeDVARS interface """
    in_file = ImageFile(mandatory = True, desc = '4D nifti input file')
    nb_time_points = traits.Int(mandatory = True, desc = 'Number of time points in the input file')
    out_file_name = traits.Str(
        mandatory = True,
        desc = 'Base name for the output file, without extension'
        )

class ComputeDVARSOutputSpec(TraitedSpec):
    """ Output specifications of a ComputeDVARS interface """
    out_file = File(
        exists = True,
        desc = 'Output file containig a regressor identifying corrupted points'
        )

class ComputeDVARS(BaseInterface):
    """ Map the MATLAB code from the following article :

    Afyouni, Soroosh & Nichols, Thomas. (2018).
    Insight and inference for DVARS. NeuroImage. 172.
    10.1016/j.neuroimage.2017.12.098.

    Code is available here:
    https://github.com/asoroosh/DVARS

    Returns

    matlab_output : capture of matlab output which may be
                    parsed by user to get computation results
    """
    input_spec = ComputeDVARSInputSpec
    output_spec = ComputeDVARSOutputSpec

    def _run_interface(self, runtime):
        """ Run the DVARS computation and identify corrupted points """

        # Compute DVARS
        dvars_output = DVARS_Calc(self.inputs.in_file)

        # Identify corrupted points
        #%   find(Stat.pvals<0.05./(T-1) & Stat.DeltapDvar>5) %print corrupted DVARS data-points
        pvalues = [e < (0.05/(nb_time_points-1)) for e in dvars_output['Inference']['Pval']]
        deltapdvar = [e > 5 for e in dvars_output['DVARS']['DeltapDvar']]

        # Write result to file
        with open(abspath(self.inputs.out_file_name + '.txt'), 'w') as file:
            file.write(tabulate([a and b for a,b in zip(pvalues, deltapdvar)]))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = abspath(self.inputs.out_file_name + '.txt')
        return outputs
