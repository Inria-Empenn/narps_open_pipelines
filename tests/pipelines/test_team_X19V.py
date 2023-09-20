"""Tests for team X19V."""
from pathlib import Path

import pandas as pd
import pytest

from narps_open.pipelines.team_X19V_new import PipelineTeamX19V, rm_smoothed_files


@pytest.fixture
def pipeline(bids_dir, tmp_path, subject_id):
    pipeline = PipelineTeamX19V(bids_dir=bids_dir, subject_list=subject_id)

    pipeline.directories.results_dir = tmp_path
    pipeline.directories.set_output_dir_with_team_id(pipeline.team_id)
    pipeline.directories.set_working_dir_with_team_id(pipeline.team_id)

    return pipeline


def test_constructor(pipeline, bids_dir):
    """Test the creation of a PipelineTeamX19V object."""
    assert pipeline.fwhm == 5.0
    assert pipeline.team_id == "X19V"
    assert pipeline.contrast_list == ["0001", "0002", "0003"]
    assert pipeline.subject_list == ["001"]
    assert pipeline.directories.dataset_dir == str(bids_dir)


def test_get_subject_infos(pipeline, events_file):
    """Test the get_session_infos method of a PipelineTeamX19V object."""
    run_info = pipeline.get_subject_infos(str(events_file))

    assert run_info[0].conditions == ["trial", "gain", "loss"]
    assert run_info[0].regressor_names is None


def test_get_parameters_file(pipeline, confounds_file):
    """Test the get_parameters_file method of a PipelineTeamX19V object."""
    parameters_file = pipeline.get_parameters_file(
        file=confounds_file,
        subject_id=pipeline.subject_list[0],
        run_id=pipeline.run_list[0],
    )

    df = pd.read_csv(parameters_file[0], sep="\t")
    assert df.shape == (452, 6)


def test_get_contrasts(pipeline):
    """Test the get_contrasts method of a PipelineTeamX19V object."""
    contrasts = pipeline.get_contrasts()

    assert contrasts[0] == ("gain", "T", ["trial", "gain", "loss"], [0, 1, 0])


def test_get_subject_level_analysis(pipeline, bids_dir, result_dir):
    """Test the get_subject_level_analysis method of a PipelineTeamX19V object.

    Only a smoke test for now.
    """
    output_dir = f"NARPS-{pipeline.team_id}-reproduced"
    working_dir = Path(f"NARPS-{pipeline.team_id}-reproduced") / "intermediate_results"

    pipeline.get_subject_level_analysis(
        exp_dir=bids_dir,
        result_dir=result_dir,
        output_dir=output_dir,
        working_dir=working_dir,
    )


def test_rm_smoothed_files(smooth_dir, subject_id, run_id, tmp_path):
    """Test the remove smoothed files function."""
    result_dir = tmp_path
    working_dir = "working_dir"

    assert smooth_dir.exists()

    rm_smoothed_files(subject_id, run_id, result_dir, working_dir)

    assert not smooth_dir.exists()
