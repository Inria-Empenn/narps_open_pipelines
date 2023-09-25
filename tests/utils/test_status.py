#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.utils.status' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_status.py
    pytest -q test_status.py -k <selected_test>
"""

from os.path import join

from requests.models import Response
from requests.exceptions import HTTPError

from pytest import mark, raises, fixture

from narps_open.utils.configuration import Configuration
from narps_open.utils.status import (
    get_teams_with_pipeline_files,
    get_opened_issues,
    PipelineStatusReport
    )

@fixture
def mock_api_issue(mocker):
    """ Create a mock GitHub API response for successful query on open issues
        (Querying the actual project would lead to non reporducible results)

        This method uses the mocker from pytest-mock to replace `requests.get`,
        which is actually imported as `get` inside the `narps_open.utils.status` module.
        Hence, we patch the `narps_open.utils.status.get` method.
    """
    response = Response()
    response.status_code = 200
    def json_func_page_0():
        return [
            {
                "html_url": "url_issue_2",
                "number": 2,
                "title" : "Issue for pipeline UK24",
                "body" : "Nothing to add here."
            },
            {
                "html_url": "url_pull_3",
                "number": 3,
                "title" : "Pull request for pipeline 2T6S",
                "pull_request" : {},
                "body" : "Work has been done."
            }
        ]

    json_func_page_1 = json_func_page_0

    def json_func_page_2():
        return [
            {
                "html_url": "url_issue_4",
                "number": 4,
                "title" : None,
                "body" : "This is a malformed issue about C88N."
            },
            {
                "html_url": "url_issue_5",
                "number": 5,
                "title" : "Issue about 2T6S",
                "body" : "Something about 2T6S."
            }
        ]

    def json_func_page_3():
        return []

    response.json = json_func
    mocker.patch('narps_open.utils.status.get', return_value = response)
    mocker.patch(
        'narps_open.utils.status.get_teams_with_pipeline_files',
        return_value = ['2T6S', 'UK24', 'Q6O0']
        )
    mocker.patch.dict(
        'narps_open.utils.status.implemented_pipelines',
        {
            '2T6S': 'PipelineTeam2T6S',
            'C88N': None,
            'Q6O0': 'PipelineTeamQ6O0',
            '1KB2': None,
            'UK24': None
        },
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

        test_pipeline = report.contents['2T6S']
        assert test_pipeline['softwares'] == 'SPM'
        assert test_pipeline['fmriprep'] == 'Yes'
        assert test_pipeline['issues'] == {5: 'url_issue_5'}
        assert test_pipeline['pulls'] == {3: 'url_pull_3'}
        assert test_pipeline['status'] == '1-progress'
        test_pipeline = report.contents['UK24']
        assert test_pipeline['softwares'] == 'SPM'
        assert test_pipeline['fmriprep'] == 'No'
        assert test_pipeline['issues'] == {2: 'url_issue_2'}
        assert test_pipeline['pulls'] == {}
        assert test_pipeline['status'] == '1-progress'
        test_pipeline = report.contents['Q6O0']
        assert test_pipeline['softwares'] == 'SPM'
        assert test_pipeline['fmriprep'] == 'Yes'
        assert test_pipeline['issues'] == {}
        assert test_pipeline['pulls'] == {}
        assert test_pipeline['status'] == '0-done'
        test_pipeline = report.contents['1KB2']
        assert test_pipeline['softwares'] == 'FSL'
        assert test_pipeline['fmriprep'] == 'No'
        assert test_pipeline['issues'] == {}
        assert test_pipeline['pulls'] == {}
        assert test_pipeline['status'] == '2-idle'
        test_pipeline = report.contents['C88N']
        assert test_pipeline['softwares'] == 'SPM'
        assert test_pipeline['fmriprep'] == 'Yes'
        assert test_pipeline['issues'] == {}
        assert test_pipeline['pulls'] == {}
        assert test_pipeline['status'] == '2-idle'

        # Test the sorting
        test_list = list(report.contents.keys())
        assert test_list[0] == 'Q6O0'
        assert test_list[1] == 'UK24'
        assert test_list[2] == '2T6S'
        assert test_list[3] == '1KB2'
        assert test_list[4] == 'C88N'

    @staticmethod
    @mark.unit_test
    def test_markdown(mock_api_issue):
        """ Test writing a PipelineStatusReport as Markdown """

        # Generate markdown from report
        report = PipelineStatusReport()
        report.generate()
        markdown = report.markdown()

        # Compare markdown with test file
        test_file_path = join(
            Configuration()['directories']['test_data'],
            'utils', 'status', 'test_markdown.md'
            )
        with open(test_file_path, 'r', encoding = 'utf-8') as file:
            assert markdown == file.read()

    @staticmethod
    @mark.unit_test
    def test_str(mock_api_issue):
        """ Test writing a PipelineStatusReport as JSON """

        # Generate report
        report = PipelineStatusReport()
        report.generate()

        # Compare string version of the report with test file
        test_file_path = join(
            Configuration()['directories']['test_data'],
            'utils', 'status', 'test_str.json'
            )
        with open(test_file_path, 'r', encoding = 'utf-8') as file:
            assert str(report) == file.read()

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
