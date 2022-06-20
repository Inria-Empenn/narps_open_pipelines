from lib.utils import return_team_config


def test_return_team_config():

    team_ID = "9Q6R"
    cfg = return_team_config(team_ID)

    assert cfg["teamID"] == "9Q6R"
    assert cfg["excluded_participants"] == ["018", "030", "088", "100"]
    assert cfg["func_fwhm"] == 5
    assert cfg["directories"]["output"] == "NARPS-9Q6R-reproduced"

    assert "018" not in cfg["subject_list"]


def test_return_team_config_default():

    team_ID = None
    cfg = return_team_config(team_ID)

    assert cfg["teamID"] is None
    assert cfg["excluded_participants"] is None
    assert cfg["func_fwhm"] is None
    assert cfg["directories"]["output"] == "NARPS-None-reproduced"
