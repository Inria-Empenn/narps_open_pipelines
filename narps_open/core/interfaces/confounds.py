#!/usr/bin/python
# coding: utf-8

""" This module defines custom interfaces related to confounds computation """

from os.path import abspath
from json import dump
from itertools import zip_longest

from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, traits, TraitedSpec, ImageFile, File
    )
from DVARS import DVARS_Calc

class ComputeDVARSInputSpec(BaseInterfaceInputSpec):
    """ Input specifications of a ComputeDVARS interface """
    in_file = ImageFile(mandatory = True, desc = '4D nifti input file')
    out_file_name = traits.Str(
        mandatory = True,
        desc = 'Base name for the output file, without extension'
        )

class ComputeDVARSOutputSpec(TraitedSpec):
    """ Output specifications of a ComputeDVARS interface """
    dvars_out_file = File(
        exists = True,
        desc = 'Output file containig DVARS results'
        )
    inference_out_file = File(
        exists = True,
        desc = 'Output file containig Inference results'
        )
    stats_out_file = File(
        exists = True,
        desc = 'Output file containig Stats results'
        )

class ComputeDVARS(BaseInterface):
    """ Map the MATLAB code from the following article :

    Afyouni, Soroosh & Nichols, Thomas. (2018).
    Insight and inference for DVARS. NeuroImage. 172.
    10.1016/j.neuroimage.2017.12.098.

    Code is available here:
    https://github.com/asoroosh/DVARS

    Outputs a file containing DVARS computation at each time point of the in_file
    """
    input_spec = ComputeDVARSInputSpec
    output_spec = ComputeDVARSOutputSpec

    def _run_interface(self, runtime):
        """ Run the DVARS computation and identify corrupted points """

        # Compute DVARS
        dvars = DVARS_Calc(self.inputs.in_file)

        # Write results to DVARS file
        with open(abspath(self.inputs.out_file_name + '_DVARS.tsv'), 'w') as file:
            # Write header
            file.write('DVARS\tDeltapDvar\tNDVARS_X2\n')
            
            # Write data
            for data in zip(
                dvars['DVARS']['DVARS'].tolist(),
                dvars['DVARS']['DeltapDvar'].tolist(),
                dvars['DVARS']['NDVARS_X2'].tolist()):
                file.write('\t'.join([str(e) for e in data]) + '\n')

        # Write results to Inference file
        with open(abspath(self.inputs.out_file_name + '_Inference.tsv'), 'w') as file:
            # Write header
            file.write('Pval\tH\tHStat\tHPrac\n')

            # Write data
            for data in zip_longest(
                dvars['Inference']['Pval'].tolist(),
                dvars['Inference']['H'].tolist(),
                dvars['Inference']['HStat'][0].tolist(),
                dvars['Inference']['HPrac'][0].tolist()):
                file.write('\t'.join([str(e) for e in data]) + '\n')

        # Write results to Stats file
        with open(abspath(self.inputs.out_file_name + '_Stats.tsv'), 'w') as file:
            # Write header
            file.write('Mean\tSD\tDF\n')

            # Write data
            file.write(f"{dvars['Stats']['Mean']}\t{dvars['Stats']['SD']}\t{dvars['Stats']['DF']}\n")

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['dvars_out_file'] = abspath(self.inputs.out_file_name + '_DVARS.tsv')
        outputs['inference_out_file'] = abspath(self.inputs.out_file_name + '_Inference.tsv')
        outputs['stats_out_file'] = abspath(self.inputs.out_file_name + '_Stats.tsv')

        return outputs
