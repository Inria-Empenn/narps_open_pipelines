"""Tests for team X19V."""
from pathlib import Path

import pandas as pd
import pytest

from narps_open.pipelines.team_X19V_new import PipelineTeamX19V


@pytest.fixture
def pipeline(bids_dir, tmp_path, subject_id):
    """Set up the pipeline with one subject and the proper directories."""
    pipeline = PipelineTeamX19V(bids_dir=bids_dir, subject_list=subject_id)

    pipeline.directories.results_dir = tmp_path
    pipeline.directories.set_output_dir_with_team_id(pipeline.team_id)
    pipeline.directories.set_working_dir_with_team_id(pipeline.team_id)

    return pipeline


@pytest.mark.unit_test
def test_constructor(pipeline, bids_dir):
    """Test the creation of a PipelineTeamX19V object."""
    assert pipeline.fwhm == 5.0
    assert pipeline.team_id == "X19V"
    assert pipeline.contrast_list == ["0001", "0002", "0003"]
    assert pipeline.subject_list == ["001"]
    assert pipeline.directories.dataset_dir == str(bids_dir)


@pytest.mark.unit_test
def test_get_subject_infos(pipeline, events_file):
    """Test the get_session_infos method of a PipelineTeamX19V object."""
    run_info = pipeline.get_subject_infos(str(events_file))

    assert run_info[0].conditions == ["trial", "gain", "loss"]
    assert run_info[0].regressor_names is None


@pytest.mark.unit_test
def test_get_parameters_file(pipeline, confounds_file):
    """Test the get_parameters_file method of a PipelineTeamX19V object."""
    parameters_file = pipeline.get_parameters_file(
        file=confounds_file,
        subject_id=pipeline.subject_list[0],
        run_id=pipeline.run_list[0],
    )

    df = pd.read_csv(parameters_file[0], sep="\t")
    assert df.shape == (452, 6)


@pytest.mark.unit_test
def test_get_contrasts(pipeline):
    """Test the get_contrasts method of a PipelineTeamX19V object."""
    contrasts = pipeline.get_contrasts()

    assert contrasts[0] == ("gain", "T", ["trial", "gain", "loss"], [0, 1, 0])


@pytest.mark.unit_test
def test_get_subject_level_analysis(pipeline):
    """Test the get_subject_level_analysis method of a PipelineTeamX19V object.

    Only a smoke test for now.
    """
    pipeline.get_subject_level_analysis()


@pytest.mark.unit_test
def test_rm_smoothed_files(pipeline):
    """Test the remove smoothed files function."""
    subject_id = pipeline.subject_list[0]
    run_id = pipeline.run_list[0]

    smooth_dir = (
        Path(pipeline.directories.results_dir)
        / pipeline.directories.working_dir
        / "l1_analysis"
        / f"_run_id_{run_id}_subject_id_{subject_id}"
        / "smooth"
    )
    smooth_dir.mkdir(parents=True, exist_ok=True)

    pipeline.rm_smoothed_files(subject_id, run_id)

    assert not smooth_dir.exists()


@pytest.mark.unit_test
@pytest.mark.parametrize(
    "method, expected",
    [
        ("equalRange", {"group_mean": [1, 1, 1]}),
        ("equalIndifference", {"group_mean": [1, 1]}),
        # TODO the following looks WRONG
        (
            "groupComp",
            {"equalRange": [0, 1, 1, 1, 1], "equalIndifference": [1, 0, 0, 0, 0]},
        ),
    ],
)
def test_get_rergressors(pipeline, method, expected):
    regressors = pipeline.get_regressors(
        equal_range_id=["002", "004", "006"],
        equal_indifference_id=["001", "003"],
        method=method,
    )
    assert regressors == expected
