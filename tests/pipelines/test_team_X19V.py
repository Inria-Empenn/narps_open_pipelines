"""Tests for team X19V."""
import pandas as pd

from narps_open.pipelines.team_X19V_new import PipelineTeamX19V, rm_smoothed_files


def test_constructor():
    """Test the creation of a PipelineTeamX19V object."""
    pipeline = PipelineTeamX19V()

    assert pipeline.fwhm == 5.0
    assert pipeline.team_id == "X19V"
    assert pipeline.contrast_list == ["0001", "0002", "0003"]


def test_get_session_infos(events_file):
    """Test the get_session_infos method of a PipelineTeamX19V object."""
    pipeline = PipelineTeamX19V()

    subject_info = pipeline.get_session_infos(str(events_file))

    assert subject_info[0].conditions == ["trial", "gain", "loss"]
    assert subject_info[0].regressor_names is None


def test_get_parameters_file(confounds_file, subject_id, run_id, tmp_path):
    """Test the get_parameters_file method of a PipelineTeamX19V object."""
    pipeline = PipelineTeamX19V()

    result_dir = tmp_path
    working_dir = "working_dir"
    (result_dir / working_dir).mkdir()

    parameters_file = pipeline.get_parameters_file(
        confounds_file, subject_id, run_id, result_dir, working_dir
    )

    df = pd.read_csv(parameters_file[0], sep="\t")
    assert df.shape == (452, 6)


def test_get_contrasts():
    """Test the get_contrasts method of a PipelineTeamX19V object."""
    pipeline = PipelineTeamX19V()

    contrasts = pipeline.get_contrasts()

    assert contrasts[0] == ("gain", "T", ["trial", "gain", "loss"], [0, 1, 0])


def test_rm_smoothed_files(smooth_dir, subject_id, run_id, tmp_path):
    """Test the remove smoothed files function."""
    result_dir = tmp_path
    working_dir = "working_dir"

    assert smooth_dir.exists()

    rm_smoothed_files(subject_id, run_id, result_dir, working_dir)

    assert not smooth_dir.exists()
