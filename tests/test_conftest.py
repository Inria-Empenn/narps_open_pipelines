#!/usr/bin/python
# coding: utf-8

""" Tests of the 'conftest.py' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_conftest.py
    pytest -q test_conftest.py -k <selected_test>
"""

from os import remove
from os.path import join, isdir, isfile

from datetime import datetime

from pytest import mark, helpers, raises

from nipype import Node, Workflow
from nipype.interfaces.utility import Function

from narps_open.utils.configuration import Configuration
from narps_open.runner import PipelineRunner
from narps_open.pipelines import Pipeline

class MockupPipeline(Pipeline):
    """ A simple Pipeline class for test purposes """

    def __init__(self, base_dir: str):
        super().__init__()
        self.base_dir = base_dir
        self.test_file = join(base_dir, 'test_conftest.txt')

        # Init the test_file : write a number of execution set to zero
        with open(self.test_file, 'w', encoding = 'utf-8') as file:
            file.write(str(0))

    def update_execution_count(_, file_path: str, workflow_name: str, nb_subjects: int):
        """ Method used inside a nipype Node, to update the execution count inside the file.
        Arguments:
        - file_path:str, path to the execution count file
        - workflow_name:str, name of the workflow
        - nb_subjects:int, number of subjects in the workflow

        Return: the updated number of executions
        """

        # Read file counter
        execution_counter = 0
        with open(file_path, 'r', encoding = 'utf-8') as file:
            # Get last char of the file
            execution_counter = int(file.read().split(' ')[-1])

        execution_counter += 1

        # Write execution count back
        with open(file_path, 'a', encoding = 'utf-8') as file:
            file.write(f'\n{workflow_name} {nb_subjects} {execution_counter}')

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
            input_names = ['_', 'file_path', 'workflow_name', 'nb_subjects'],
            output_names = ['execution_counter'],
            function = self.update_execution_count),
            name = 'node_count'
            )
        # this input is set to now(), so that it changes at every run, thus preventing
        # nipype's cache to work
        node_count.inputs._ = datetime.now()
        node_count.inputs.file_path = self.test_file
        node_count.inputs.workflow_name = workflow_name
        node_count.inputs.nb_subjects = len(self.subject_list)

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
            base_dir = self.base_dir,
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
        template = join(self.base_dir, 'subject_id_{subject_id}_output_preprocessing_1.md')
        return [template.format(subject_id = s) for s in self.subject_list]

    def get_run_level_outputs(self):
        """ Return a list of templates of the output files generated by the run level analysis.
            Templates are expressed relatively to the self.directories.output_dir.
        """
        template = join(self.base_dir, 'subject_id_{subject_id}_output_run_1.md')
        return [template.format(subject_id = s) for s in self.subject_list]

    def get_subject_level_outputs(self):
        """ Return a list of templates of the output files generated by the subject level analysis.
            Templates are expressed relatively to the self.directories.output_dir.
        """
        template = join(self.base_dir, 'subject_id_{subject_id}_output_analysis_1.md')
        return [template.format(subject_id = s) for s in self.subject_list]

    def get_group_level_outputs(self):
        """ Return a list of templates of the output files generated by the group level analysis.
            Templates are expressed relatively to the self.directories.output_dir.
        """
        templates = [
            join(self.base_dir, 'group_{nb_subjects}_output_a.md'),
            join(self.base_dir, 'group_{nb_subjects}_output_b.md')
            ]
        return_list = [t.format(nb_subjects = len(self.subject_list)) for t in  templates]

        template = join(self.base_dir, 'hypothesis_{id}.md')
        return_list += [template.format(id = i) for i in range(1,19)]

        return return_list

    def get_hypotheses_outputs(self):
        """ Return the names of the files used by the team to answer the hypotheses of NARPS. """
        template = join(self.base_dir, 'hypothesis_{id}.md')
        return [template.format(id = i) for i in range(1,19)]

class MockupResultsCollection():
    """ A fake ResultsCollection object for test purposes """

    def __init__(self, team_id: str):
        self.team_id = team_id
        self.uid = self.get_uid()
        self.directory = join(
            Configuration()['directories']['narps_results'],
            'orig',
            self.uid + '_' + self.team_id
            )
        self.files = self.get_file_urls()

    def get_uid(self):
        """ Return the uid of the collection by browsing the team description """
        return 'uid'

    def get_file_urls(self):
        """ Return a dict containing the download url for each file of the collection.
        * dict key is the file base name (with extension)
        * dict value is the download url for the file on Neurovault
        """
        urls = {}
        for file_id in range(1, 19):
            urls[f'file_{file_id}'] = 'url'

        return urls

    def download(self):
        """ Download the collection, file by file. """

class TestConftest:
    """ A class that contains all the unit tests for the conftest module."""

    @staticmethod
    @mark.unit_test
    def test_compare_float_2d_arrays():
        """ Test the compare_float_2d_arrays helper """

        array_1 = [[5.0, 0.0], [1.0, 2.0]]
        array_2 = [[5.0, 0.0], [1.0]]
        with raises(AssertionError):
            helpers.compare_float_2d_arrays(array_1, array_2)

        array_1 = [[6.0, 0.0], [1.0]]
        array_2 = [[6.0, 0.0], [1.0, 2.0]]
        with raises(AssertionError):
            helpers.compare_float_2d_arrays(array_1, array_2)

        array_1 = [[7.10001, 0.0], [1.0, 2.0]]
        array_2 = [[7.10001, 0.0], [1.0, 2.00003]]
        with raises(AssertionError):
            helpers.compare_float_2d_arrays(array_1, array_2)

        array_1 = [[10.0000200, 15.10], [1.0, 2.0]]
        array_2 = [[10.00002, 15.10000], [1.0, 2.000003]]
        helpers.compare_float_2d_arrays(array_1, array_2)

    @staticmethod
    @mark.unit_test
    def test_test_outputs(temporary_data_dir):
        """ Test the test_pipeline_outputs helper """

        # Test pipeline
        pipeline = MockupPipeline(temporary_data_dir)
        pipeline.subject_list = ['001', '002']

        # Wrong length for nb_of_outputs
        with raises(AssertionError):
            helpers.test_pipeline_outputs(pipeline, [1,2,3])

        # Wrong number of outputs
        with raises(AssertionError):
            helpers.test_pipeline_outputs(pipeline, [0, 2, 2, 20, 18])
        with raises(AssertionError):
            helpers.test_pipeline_outputs(pipeline, [2, 0, 2, 20, 18])
        with raises(AssertionError):
            helpers.test_pipeline_outputs(pipeline, [2, 2, 0, 20, 18])
        with raises(AssertionError):
            helpers.test_pipeline_outputs(pipeline, [2, 2, 2, 0, 18])
        with raises(AssertionError):
            helpers.test_pipeline_outputs(pipeline, [2, 2, 2, 20, 0])

        # Right number of outputs
        helpers.test_pipeline_outputs(pipeline, [2, 2, 2, 20, 18])

        # Not a valid path name
        pipeline.get_group_level_outputs = lambda : 'not_fo\rmatted'
        with raises(AssertionError):
            helpers.test_pipeline_outputs(pipeline, [2, 2, 2, 1, 18])

        # Not a valid path name
        pipeline.get_group_level_outputs = lambda : '{not_formatted'
        with raises(AssertionError):
            helpers.test_pipeline_outputs(pipeline, [2, 2, 2, 1, 18])

        # Not a valid path name
        pipeline.get_group_level_outputs = lambda : '{not_formatted'
        with raises(AssertionError):
            helpers.test_pipeline_outputs(pipeline, [2, 2, 2, 1, 18])

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
    def test_test_pipeline_execution(mocker, temporary_data_dir):
        """ Test the test_pipeline_execution helper """

        # Set subgroups of subjects
        Configuration()['testing']['pipelines']['nb_subjects_per_group'] = 4

        # Create mocks
        mocker.patch('conftest.get_correlation_coefficient', return_value = 1.0)
        fake_runner = PipelineRunner('2T6S')
        fake_runner._pipeline = MockupPipeline(temporary_data_dir)
        mocker.patch('conftest.PipelineRunner', return_value = fake_runner)
        mocker.patch('conftest.ResultsCollection', return_value = MockupResultsCollection('2T6S'))

        # Run pipeline
        helpers.test_pipeline_execution('test_conftest', 7)

        # Check outputs
        assert isdir(join(temporary_data_dir, 'TestConftest_preprocessing_workflow'))
        assert isdir(join(temporary_data_dir, 'TestConftest_run_level_workflow'))
        assert isdir(join(temporary_data_dir, 'TestConftest_subject_level_workflow'))
        assert isdir(join(temporary_data_dir, 'TestConftest_group_level_workflow'))

        # Check executions
        with open(join(temporary_data_dir, 'test_conftest.txt'), 'r', encoding = 'utf-8') as file:
            assert file.readline() == '0\n'
            # First exec of preprocessing creates an exception (execution counter == 1)
            assert file.readline() == 'TestConftest_preprocessing_workflow 4 1\n'
            # Relaunching the workflow
            # Preprocessing files won't be created(execution counter == 2)
            assert file.readline() == 'TestConftest_preprocessing_workflow 4 2\n'
            assert file.readline() == 'TestConftest_run_level_workflow 4 3\n'
            assert file.readline() == 'TestConftest_subject_level_workflow 4 4\n'
            # Relaunching the workflow
            # Everything's fine
            assert file.readline() == 'TestConftest_preprocessing_workflow 4 5\n'
            assert file.readline() == 'TestConftest_run_level_workflow 4 6\n'
            assert file.readline() == 'TestConftest_subject_level_workflow 4 7\n'
            assert file.readline() == 'TestConftest_preprocessing_workflow 3 8\n'
            assert file.readline() == 'TestConftest_run_level_workflow 3 9\n'
            assert file.readline() == 'TestConftest_subject_level_workflow 3 10\n'
            assert file.readline() == 'TestConftest_group_level_workflow 7 11'

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
