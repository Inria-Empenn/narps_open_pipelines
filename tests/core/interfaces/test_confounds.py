#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.core.interfaces.confounds' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_confounds.py
    pytest -q test_confounds.py -k <selected_test>
"""
from os.path import join, exists
from filecmp import cmp

from pytest import mark
from nipype import Node

from narps_open.core.interfaces import confounds
from narps_open.utils.configuration import Configuration

class TestDvars:
    """ A class that contains all the unit tests for the DVARS interface."""

    @staticmethod
    @mark.unit_test
    def test_run(temporary_data_dir):
        """ Test to run the interface """

        # Get input file
        test_data_path = join(Configuration()['directories']['test_data'],
            'core', 'interfaces', 'confounds')
        test_file_path = join(test_data_path, 'dvars_in_file.nii')

        # Run dvars
        dvars = Node(confounds.ComputeDVARS(), name = 'dvars')
        dvars.inputs.in_file = test_file_path
        dvars.inputs.out_file_name = 'dvars_results'
        dvars.base_dir = temporary_data_dir
        dvars.run()

        # Check output files
        assert exists(join(temporary_data_dir, 'dvars', 'dvars_results_DVARS.tsv'))
        assert exists(join(temporary_data_dir, 'dvars', 'dvars_results_Inference.tsv'))
        assert exists(join(temporary_data_dir, 'dvars', 'dvars_results_Stats.tsv'))
        assert cmp(
            join(temporary_data_dir, 'dvars', 'dvars_results_DVARS.tsv'),
            join(test_data_path, 'dvars_results_DVARS.tsv')
            )
        assert cmp(
            join(temporary_data_dir, 'dvars', 'dvars_results_Inference.tsv'),
            join(test_data_path, 'dvars_results_Inference.tsv')
            )
        assert cmp(
            join(temporary_data_dir, 'dvars', 'dvars_results_Stats.tsv'),
            join(test_data_path, 'dvars_results_Stats.tsv')
            )
