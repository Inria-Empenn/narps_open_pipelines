#!/usr/bin/python
# coding: utf-8

""" Tests of the 'conftest.py' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_conftest.py
    pytest -q test_conftest.py -k <selected_test>
"""

from os import makedirs, remove
from os.path import join, abspath, isdir, isfile
from shutil import rmtree

from datetime import datetime

from pytest import mark, helpers, fixture

from nipype import Node, Workflow
from nipype.interfaces.utility import Function

from narps_open.utils.configuration import Configuration
from narps_open.runner import PipelineRunner
from narps_open.pipelines import Pipeline
from narps_open.data.results import ResultsCollection

TEST_DIR = abspath(join(Configuration()['directories']['test_runs'], 'test_conftest'))

@fixture
def set_test_directory(scope = 'function'):
    rmtree(TEST_DIR, ignore_errors = True)
    makedirs(TEST_DIR, exist_ok = True)

    yield

    # Comment this line for debugging
    #rmtree(TEST_DIR, ignore_errors = True)

class MockupPipeline(Pipeline):
    """ A simple Pipeline class for test purposes """

    def __init__(self):
        super().__init__()
        self.test_file = join(TEST_DIR, 'test_conftest.txt')

        # Init the test_file : write a number of execution set to zero
        with open(self.test_file, 'w', encoding = 'utf-8') as file:
            file.write(str(0))

    def update_execution_count(_, file_path: str, workflow_name: str):
        """ Method used inside a nipype Node, to update the execution count inside the file.
        Arguments:
        - file_path:str, path to the execution count file
        - workflow_name:str, name of the workflow

        Return: the updated number of executions
        """

        # Read file counter
        execution_counter = 0
        with open(file_path, 'r', encoding = 'utf-8') as file:
            # Get last char of the file
            execution_counter = int(file.read()[-1])

        execution_counter += 1

        # Write execution count back
        with open(file_path, 'a', encoding = 'utf-8') as file:
            file.write(f'\n{workflow_name} {execution_counter}')

        return execution_counter

    def decide_exception(execution_counter: int):
        """ Method used inside a nipype Node, to simulate an exception during one run """
        if execution_counter == 1:
            raise AttributeError

    def write_files(_, file_list: list, execution_counter: int):
        """ Method used inside a nipype Node, to create a set of files """
        from pathlib import Path

        if execution_counter != 2:
            for file_path in file_list:
                Path(file_path).touch()

    def create_workflow(self, workflow_name: str, file_list: list):
        """ Return a nipype workflow with two nodes.
            First node updates the number of executions of the workflow?
            Second node produces an exception, every two executions of the workflow.
            Third node writes the output files, except once every three executions
                of the workflow.

            Arguments:
            - workflow_name: str, the name of the workflow to create
            - file_list: list, list of the files that the workflow is supposed to generate
        """
        node_count = Node(Function(
            input_names = ['_', 'file_path', 'workflow_name'],
            output_names = ['execution_counter'],
            function = self.update_execution_count),
            name = 'node_count'
            )
        # this input is set to now(), so that it changes at every run, thus preventing
        # nipype's cache to work
        node_count.inputs._ = datetime.now()
        node_count.inputs.file_path = self.test_file
        node_count.inputs.workflow_name = workflow_name

        node_decide = Node(Function(
            input_names = ['execution_counter'],
            output_names = ['_'],
            function = self.decide_exception),
            name = 'node_decide'
            )

        node_files = Node(Function(
            input_names = ['_', 'file_list', 'execution_counter'],
            output_names = [],
            function = self.write_files),
            name = 'node_files'
            )
        node_files.inputs.file_list = file_list

        workflow = Workflow(
            base_dir = TEST_DIR,
            name = workflow_name
            )
        workflow.add_nodes([node_count, node_decide, node_files])
        workflow.connect(node_count, 'execution_counter', node_decide, 'execution_counter')
        workflow.connect(node_count, 'execution_counter', node_files, 'execution_counter')
        workflow.connect(node_decide, '_', node_files, '_')

        return workflow

    def get_preprocessing(self):
        """ Return a fake preprocessing workflow """
        return self.create_workflow(
            'TestConftest_preprocessing_workflow',
            self.get_preprocessing_outputs()
            )

    def get_run_level_analysis(self):
        """ Return a fake run level workflow """
        return self.create_workflow(
            'TestConftest_run_level_workflow',
            self.get_run_level_outputs()
            )

    def get_subject_level_analysis(self):
        """ Return a fake subject level workflow """
        return self.create_workflow(
            'TestConftest_subject_level_workflow',
            self.get_subject_level_outputs()
            )

    def get_group_level_analysis(self):
        """ Return a fake subject level workflow """
        return self.create_workflow(
            'TestConftest_group_level_workflow',
            self.get_group_level_outputs()
            )

    def get_preprocessing_outputs(self):
        """ Return a list of templates of the output files generated by the preprocessing """
        template = join(TEST_DIR, 'subject_id_{subject_id}_output_preprocessing_1.md')
        return [template.format(subject_id = s) for s in self.subject_list]

    def get_run_level_outputs(self):
        """ Return a list of templates of the output files generated by the run level analysis.
            Templates are expressed relatively to the self.directories.output_dir.
        """
        template = join(TEST_DIR, 'subject_id_{subject_id}_output_run_1.md')
        return [template.format(subject_id = s) for s in self.subject_list]

    def get_subject_level_outputs(self):
        """ Return a list of templates of the output files generated by the subject level analysis.
            Templates are expressed relatively to the self.directories.output_dir.
        """
        template = join(TEST_DIR, 'subject_id_{subject_id}_output_analysis_1.md')
        return [template.format(subject_id = s) for s in self.subject_list]

    def get_group_level_outputs(self):
        """ Return a list of templates of the output files generated by the group level analysis.
            Templates are expressed relatively to the self.directories.output_dir.
        """
        templates = [
            join(TEST_DIR, 'group_{nb_subjects}_output_a.md'),
            join(TEST_DIR, 'group_{nb_subjects}_output_b.md')
            ]
        return_list = [t.format(nb_subjects = len(self.subject_list)) for t in  templates]

        template = join(TEST_DIR, 'hypothesis_{id}.md')
        return_list += [template.format(id = i) for i in range(1,19)]

        return return_list

    def get_hypotheses_outputs(self):
        """ Return the names of the files used by the team to answer the hypotheses of NARPS. """
        template = join(TEST_DIR, 'hypothesis_{id}.md')
        return [template.format(id = i) for i in range(1,19)]

class TestConftest:
    """ A class that contains all the unit tests for the conftest module."""

    @staticmethod
    @mark.unit_test
    def test_test_correlation_results(mocker):
        """ Test the test_correlation_result helper """

        mocker.patch(
            'conftest.Configuration',
            return_value = {
                'testing': {
                    'pipelines': {
                        'correlation_thresholds' : [0.30, 0.70, 0.80, 0.85, 0.93]
                    }
                }
            }
        )

        assert helpers.test_correlation_results(
            [0.35, 0.35, 0.36, 0.37, 0.38, 0.39, 0.99, 0.82, 0.40], 20)
        assert helpers.test_correlation_results(
            [0.75, 0.75, 0.76, 0.77, 0.88, 0.79, 0.99, 0.82, 0.80], 40)
        assert helpers.test_correlation_results(
            [0.85, 0.85, 0.86, 0.87, 0.98, 0.99, 0.99, 0.82, 0.81], 60)
        assert helpers.test_correlation_results(
            [0.86, 0.86, 0.86, 0.87, 0.88, 0.89, 0.99, 0.92, 0.95], 80)
        assert helpers.test_correlation_results(
            [0.95, 0.95, 0.96, 0.97, 0.98, 0.99, 0.99, 0.95, 1.0], 108)
        assert not helpers.test_correlation_results(
            [0.3, 0.29, 0.3, 0.3, 0.3, 0.39, 0.99, 0.82, 0.40], 20)
        assert not helpers.test_correlation_results(
            [0.60, 0.7, 0.7, 0.7, 0.8, 0.79, 0.99, 0.82, 0.80], 40)
        assert not helpers.test_correlation_results(
            [0.8, 0.79, 0.8, 0.8, 0.9, 0.99, 0.99, 0.82, 0.81], 60)
        assert not helpers.test_correlation_results(
            [0.8, 0.8, 0.8, 0.8, 0.88, 0.89, 0.99, 0.92, 0.95], 80)
        assert not helpers.test_correlation_results(
            [0.99, 0.99, 0.99, 1., 1., 1., -1., 0.95, 1.0], 108)

    @staticmethod
    @mark.unit_test
    def test_test_pipeline_execution(mocker, set_test_directory):
        """ Test the test_pipeline_execution helper """

        # Create mocks
        mocker.patch('conftest.get_correlation_coefficient', return_value = 1.0)
        fake_runner = PipelineRunner('2T6S')
        fake_runner._pipeline = MockupPipeline()
        mocker.patch('conftest.PipelineRunner', return_value = fake_runner)
        fake_collection = ResultsCollection('2T6S')
        mocker.patch('conftest.ResultsCollection', return_value = fake_collection)

        # Run pipeline
        helpers.test_pipeline_execution('test_conftest', 20)

        # Check outputs
        assert isdir(join(TEST_DIR, 'TestConftest_preprocessing_workflow'))
        assert isdir(join(TEST_DIR, 'TestConftest_run_level_workflow'))
        assert isdir(join(TEST_DIR, 'TestConftest_subject_level_workflow'))
        assert isdir(join(TEST_DIR, 'TestConftest_group_level_workflow'))

        # Check executions
        with open(join(TEST_DIR, 'test_conftest.txt'), 'r', encoding = 'utf-8') as file:
            assert file.readline() == '0\n'
            # First exec of preprocessing creates an exception (execution counter == 1)
            assert file.readline() == 'TestConftest_preprocessing_workflow 1\n'
            # Relaunching the workflow
            # Preprocessing files won't be created(execution counter == 2)
            assert file.readline() == 'TestConftest_preprocessing_workflow 2\n'
            assert file.readline() == 'TestConftest_run_level_workflow 3\n'
            assert file.readline() == 'TestConftest_subject_level_workflow 4\n'
            # Relaunching the workflow
            # Everything's fine
            assert file.readline() == 'TestConftest_preprocessing_workflow 5\n'
            assert file.readline() == 'TestConftest_run_level_workflow 6\n'
            assert file.readline() == 'TestConftest_subject_level_workflow 7\n'
            assert file.readline() == 'TestConftest_group_level_workflow 8'

    @staticmethod
    @mark.unit_test
    def test_test_pipeline_evaluation(mocker):
        """ Test the test_pipeline_evaluation helper """

        # Create mocks
        mocker.patch('conftest.test_pipeline_execution',
            return_value = [0.1, 0.2, 0.3, 0.4, 0.55555, 0.6, 0.7, 0.8, 0.999999]
            )
        mocker.patch('conftest.test_correlation_results', return_value = True)

        # Run helper
        helpers.test_pipeline_evaluation('fake_team_id')

        assert isfile('test_pipeline-fake_team_id.txt')

        with open('test_pipeline-fake_team_id.txt', 'r', encoding = 'utf-8') as file:
            file_contents = file.read()

        remove('test_pipeline-fake_team_id.txt')

        check_file_contents = '| fake_team_id | 20 subjects | success '
        check_file_contents += '| [0.1, 0.2, 0.3, 0.4, 0.56, 0.6, 0.7, 0.8, 1.0] |\n'
        check_file_contents += '| fake_team_id | 40 subjects | success '
        check_file_contents += '| [0.1, 0.2, 0.3, 0.4, 0.56, 0.6, 0.7, 0.8, 1.0] |\n'
        check_file_contents += '| fake_team_id | 60 subjects | success '
        check_file_contents += '| [0.1, 0.2, 0.3, 0.4, 0.56, 0.6, 0.7, 0.8, 1.0] |\n'
        check_file_contents += '| fake_team_id | 80 subjects | success '
        check_file_contents += '| [0.1, 0.2, 0.3, 0.4, 0.56, 0.6, 0.7, 0.8, 1.0] |\n'
        check_file_contents += '| fake_team_id | 108 subjects | success '
        check_file_contents += '| [0.1, 0.2, 0.3, 0.4, 0.56, 0.6, 0.7, 0.8, 1.0] |\n'

        assert check_file_contents == file_contents
