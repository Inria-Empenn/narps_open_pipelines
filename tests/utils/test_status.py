#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.utils.status' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_status.py
    pytest -q test_status.py -k <selected_test>
"""

from requests.models import Response
from requests.exceptions import HTTPError

from pytest import mark, raises, fixture

from narps_open.utils.status import (
    get_teams_with_pipeline_files,
    get_opened_issues,
    PipelineStatusReport
    )

@fixture
def mock_api_issue(mocker):
    """ Create a mock GitHub API response for successful querry on open issues
        (Querying the actual project would lead to non reporducible results)

        This method uses the mocker from pytest-mock to replace `requests.get`,
        which is actually imported as `get` inside the `narps_open.utils.status` module.
        Hence, we patch the `narps_open.utils.status.get` method.
    """
    response = Response()
    response.status_code = 200
    def json_func():
        return [
            {
                "html_url": "urls",
                "number": 2,
                "title" : "Issue for pipeline C88N",
                "body" : "Nothing to add here."
            }
        ]
    response.json = json_func
    mocker.patch('narps_open.utils.status.get', return_value = response)
    mocker.patch(
        'narps_open.utils.status.get_teams_with_pipeline_files',
        return_value = ['2T6S', 'C88N']
        )
    mocker.patch.dict(
        'narps_open.utils.status.implemented_pipelines',
        {'2T6S': 'PipelineTeam2T6S', 'C88N': None, 'Q6O0': None},
        clear = True
        )

class TestUtilsStatus:
    """ A class that contains all the unit tests for the status module."""

    @staticmethod
    @mark.unit_test
    def test_get_issues(mocker):
        """ Test the get_opened_issues function

            This method uses the mocker from pytest-mock to replace `requests.get`,
            which is actually imported as `get` inside the `narps_open.utils.status` module.
            Hence, we patch the `narps_open.utils.status.get` method.
        """
        get_opened_issues()

        # Create a mock API response for 404 error
        response = Response()
        response.status_code = 404

        mocker.patch('narps_open.utils.status.get', return_value = response)
        with raises(HTTPError):
            get_opened_issues()

        # Create a mock API response for the no issues case
        response = Response()
        response.status_code = 200
        def json_func():
            return []
        response.json = json_func
        mocker.patch('narps_open.utils.status.get', return_value = response)
        assert len(get_opened_issues()) == 0

        # Create a mock API response for the general usecase
        response = Response()
        response.status_code = 200
        def json_func():
            return [
                {
                    "html_url": "urls",
                    "number": 2,
                }
            ]
        response.json = json_func

        mocker.patch('narps_open.utils.status.get', return_value = response)
        issues = get_opened_issues()
        assert len(issues) == 1
        assert issues[0]['html_url'] == 'urls'
        assert issues[0]['number'] == 2

    @staticmethod
    @mark.unit_test
    def test_get_teams():
        """ Test the get_teams_with_pipeline_files function """

        files = get_teams_with_pipeline_files()
        assert '2T6S' in files
        assert 'C88N' in files
        assert 'J7F9' in files
        assert '43FJ' in files

    @staticmethod
    @mark.unit_test
    def test_generate(mock_api_issue):
        """ Test generating a PipelineStatusReport """

        # Test the generation
        report = PipelineStatusReport()
        report.generate()
        print(report)
        test_pipeline = report.contents['2T6S']
        assert test_pipeline['softwares'] == 'SPM'
        assert test_pipeline['fmriprep'] == 'Yes'
        assert test_pipeline['issues'] == {}
        assert test_pipeline['status'] == 'done'
        test_pipeline = report.contents['C88N']
        assert test_pipeline['softwares'] == 'SPM'
        assert test_pipeline['fmriprep'] == 'Yes'
        assert test_pipeline['issues'] == {2: 'urls'}
        assert test_pipeline['status'] == 'progress'
        test_pipeline = report.contents['Q6O0']
        assert test_pipeline['softwares'] == 'SPM'
        assert test_pipeline['fmriprep'] == 'Yes'
        assert test_pipeline['issues'] == {}
        assert test_pipeline['status'] == 'idle'

    @staticmethod
    @mark.unit_test
    def test_markdown(mock_api_issue):
        """ Test writing a PipelineStatusReport as Markdown """

        report = PipelineStatusReport()
        report.generate()
        markdown = report.markdown()
        assert '| 2T6S | :green_circle: | SPM | Yes |  |' in markdown
        assert '| C88N | :orange_circle: | SPM | Yes | [2](urls),  |' in markdown
        assert '| Q6O0 | :red_circle: | SPM | Yes |  |' in markdown

    @staticmethod
    @mark.unit_test
    def test_str(mock_api_issue):
        """ Test writing a PipelineStatusReport as JSON """
        report = PipelineStatusReport()
        report.generate()
        print(report)
        test_string = '    "2T6S": {\n        "softwares": "SPM",\n        "fmriprep": "Yes",'
        test_string += '\n        "issues": {},\n        "status": "done"\n    },'
        assert test_string in str(report)
        test_string = '    "C88N": {\n        "softwares": "SPM",\n        "fmriprep": "Yes",'
        test_string += '\n        "issues": {\n            "2": "urls"\n        },'
        test_string += '\n        "status": "progress"\n    },'
        assert test_string in str(report)

    @staticmethod
    @mark.unit_test
    def test_errors(mocker):
        """ Test error cases in a PipelineStatusReport """

        # Replace functions
        mocker.patch(
            'narps_open.utils.status.get_opened_issues',
            return_value = {}
            )
        mocker.patch(
            'narps_open.utils.status.get_teams_with_pipeline_files',
            return_value = ['C88N']
            )
        mocker.patch(
            'narps_open.pipelines.implemented_pipelines',
            return_value = {'2T6S': 'PipelineTeam2T6S', 'C88N': None}
            )

        # Test a report
        report = PipelineStatusReport()
        with raises(AttributeError):
            report.generate()
