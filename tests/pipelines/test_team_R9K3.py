#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_R9K3' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_R9K3.py
    pytest -q test_team_R9K3.py -k <selected_test>
"""

from nipype import Workflow

from narps_open.pipelines.team_R9K3 import PipelineTeamR9K3

def test_get_preprocessing():
    pipeline = PipelineTeamR9K3()
    assert pipeline.fwhm == 6
    assert pipeline.team_id == 'R9K3'
    preprocessing = pipeline.get_preprocessing()
    assert isinstance(preprocessing, Workflow)

def test_get_run_level_analysis():
    pipeline = PipelineTeamR9K3()
    assert pipeline.get_run_level_analysis() is None

def test_get_subject_infos():

    pipeline = PipelineTeamR9K3()
    pipeline.get_subject_infos(event_files=["/home/remi/github/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-01_events.tsv"], 
                               runs = [1])