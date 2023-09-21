#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.pipelines.team_R9K3' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_team_R9K3.py
    pytest -q test_team_R9K3.py -k <selected_test>
"""

from __future__ import annotations

from narps_open.data.description import TeamDescription



from nipype import Workflow
from pathlib import Path
import pytest

from narps_open.pipelines.team_R9K3 import PipelineTeamR9K3


TEAM_ID = "R9K3"
DESC = TeamDescription(TEAM_ID)

@pytest.fixture
def root_dir() -> Path:
    return Path(__file__).parent.parent.parent


@pytest.fixture
def bids_dir(root_dir) -> Path:
    return root_dir / "data" / "original" / "ds001734"


@pytest.fixture
def result_dir(root_dir):
    return root_dir / "derived" / "reproduced"


@pytest.fixture
def subject_id() -> str:
    return "001"


@pytest.fixture
def run_id() -> str:
    return "01"


@pytest.fixture
def events_file(bids_dir, subject_id, run_id) -> Path:
    """Return path to an events file from frmiprep."""
    return (
        bids_dir
        / f"sub-{subject_id}"
        / "func"
        / f"sub-{subject_id}_task-MGT_run-{run_id}_events.tsv"
    )



@pytest.fixture
def pipeline(bids_dir, tmp_path):
    """Set up the pipeline with one subject and the proper directories."""
    pipeline = PipelineTeamR9K3(bids_dir=bids_dir)

    pipeline.directories.results_dir = tmp_path
    pipeline.directories.set_output_dir_with_team_id(pipeline.team_id)
    pipeline.directories.set_working_dir_with_team_id(pipeline.team_id)

    return pipeline

def test_init(pipeline):
    assert pipeline.fwhm == 6
    assert pipeline.team_id == "R9K3"
    assert len(pipeline.subject_list) == int(DESC.derived["n_participants"])

def test_get_preprocessing(pipeline):
    preprocessing = pipeline.get_preprocessing()
    assert isinstance(preprocessing, Workflow)


def test_get_run_level_analysis(pipeline):
    assert pipeline.get_run_level_analysis() is None


# def test_get_subject_infos(pipeline):
#     pipeline.get_subject_infos(
#         event_files=[
#             "/home/remi/github/narps_open_pipelines/data/original/ds001734/sub-001/func/sub-001_task-MGT_run-01_events.tsv"
#         ],
#         runs=[1],
#     )
