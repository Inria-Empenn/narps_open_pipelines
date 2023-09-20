from pathlib import Path

from narps_open.pipelines.team_X19V_new import PipelineTeamX19V


def test_constructor():
    pipeline = PipelineTeamX19V()

    assert pipeline.fwhm == 5.0
    assert pipeline.team_id == "X19V"
    assert pipeline.contrast_list == ["0001", "0002", "0003"]


def test_get_session_infos(events_file):
    pipeline = PipelineTeamX19V()

    subject_info = pipeline.get_session_infos(str(events_file))

    assert subject_info[0].conditions == ['trial', 'gain', 'loss']
    assert subject_info[0].regressor_names is None
