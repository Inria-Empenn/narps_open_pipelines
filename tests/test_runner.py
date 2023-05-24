#!/usr/bin/python
# coding: utf-8

""" Tests of the 'runner' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_runner.py
    pytest -q test_runner.py -k <selected_test>
"""

from os import remove
from os.path import join, isfile, abspath

from datetime import datetime

from pytest import raises, mark

from nipype import Node, Workflow
from nipype.interfaces.utility import Function

from narps_open.utils.configuration import Configuration
from narps_open.runner import PipelineRunner
from narps_open.pipelines import Pipeline
from narps_open.pipelines.team_2T6S import PipelineTeam2T6S

class MockupPipeline(Pipeline):
    """ A simple Pipeline class for test purposes """

    def __init__(self):
        super().__init__()
        self.test_file = abspath(
            join(Configuration()['directories']['test_runs'], 'test_runner.txt'))
        if isfile(self.test_file):
            remove(self.test_file)

    def __del__(self):
        if isfile(self.test_file):
            remove(self.test_file)

    def write_to_file(_, text_to_write: str, file_path: str):
        """ Method used inside a nipype Node, to write a line in a test file """
        with open(file_path, 'a', encoding = 'utf-8') as file:
            file.write(text_to_write)

    def create_workflow(self, workflow_name: str):
        """ Return a nipype worflow with two nodes writing in a file """
        node_1 = Node(Function(
            input_names = ['_', 'text_to_write', 'file_path'],
            output_names = ['_'],
            function = self.write_to_file),
            name = 'node_1'
            )
        # this input is set to now(), so that it changes at every run, thus preventing
        # nipype's cache to work
        node_1.inputs._ = datetime.now()
        node_1.inputs.text_to_write = 'MockupPipeline : '+workflow_name+' node_1\n'
        node_1.inputs.file_path = self.test_file

        node_2 = Node(Function(
            input_names = ['_', 'text_to_write', 'file_path'],
            output_names = [],
            function = self.write_to_file),
            name = 'node_2'
            )
        node_2.inputs.text_to_write = 'MockupPipeline : '+workflow_name+' node_2\n'
        node_2.inputs.file_path = self.test_file

        workflow = Workflow(
            base_dir = Configuration()['directories']['test_runs'],
            name = workflow_name
            )
        workflow.add_nodes([node_1, node_2])
        workflow.connect(node_1, '_', node_2, '_')

        return workflow

    def get_preprocessing(self):
        """ Return a fake preprocessing workflow """
        return self.create_workflow('TestPipelineRunner_preprocessing_workflow')

    def get_run_level_analysis(self):
        """ Return a fake run level workflow """
        return self.create_workflow('TestPipelineRunner_run_level_workflow')

    def get_subject_level_analysis(self):
        """ Return a fake subject level workflow """
        return self.create_workflow('TestPipelineRunner_subject_level_workflow')

    def get_group_level_analysis(self):
        """ Return a fake subject level workflow """
        return self.create_workflow('TestPipelineRunner_group_level_workflow')

class MockupWrongPipeline(Pipeline):
    """ A simple Pipeline class for test purposes """

    def __init__(self):
        super().__init__()

    def get_preprocessing(self):
        return 'Wrong_workflow_type'

    def get_run_level_analysis(self):
        return None

    def get_subject_level_analysis(self):
        return None

    def get_group_level_analysis(self):
        return None

class MockupWrongPipeline2(Pipeline):
    """ A simple Pipeline class for test purposes """

    def __init__(self):
        super().__init__()

    def get_preprocessing(self):
        return ['Wrong_workflow_type', 'Wrong_workflow_type']

    def get_run_level_analysis(self):
        return None

    def get_subject_level_analysis(self):
        return None

    def get_group_level_analysis(self):
        return None

class TestPipelineRunner:
    """ A class that contains all the unit tests for the PipelineRunner class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineRunner object """

        # 1 - Instanciate a runner without team id
        with raises(KeyError):
            PipelineRunner()

        # 2 - Instanciate a runner with wrong team id
        with raises(KeyError):
            PipelineRunner('wrong_id')

        # 3 - Instanciate a runner with a not implemented team id
        with raises(NotImplementedError):
            PipelineRunner('08MQ')

        # 4 - Instanciate a runner with an implemented team id
        runner = PipelineRunner('2T6S')
        assert isinstance(runner._pipeline, PipelineTeam2T6S)
        assert runner.team_id == '2T6S'

        # 5 - Modify team id for an existing runner (with a not implemented team id)
        with raises(NotImplementedError):
            runner.team_id = '08MQ'

    @staticmethod
    @mark.unit_test
    def test_subjects():
        """ Test the PipelineRunner features of building subject lists """
        runner = PipelineRunner('2T6S')

        # 1 - random list
        # Check the right number of subjects is generated
        runner.random_nb_subjects = 8
        assert len(runner.subjects) == 8
        runner.random_nb_subjects = 4
        assert len(runner.subjects) == 4

        # Check formatting and consistency
        for subject in runner.subjects:
            assert isinstance(subject, str)
            assert len(subject) == 3
            assert int(subject) > 0
            assert int(subject) < 125

        # 2 - fixed list
        # Check subject ids too high
        with raises(AttributeError):
            runner.subjects = ['125', '043']
        with raises(AttributeError):
            runner.subjects = ['120', '048']

        # Check duplicate subject ids are removed
        runner.subjects = ['043', '022', '022', '045']
        assert runner.subjects == ['043', '022', '045']

        # Check formatting
        runner.subjects = [22, '00022', '0043', 45]
        assert runner.subjects == ['022', '043', '045']

    @staticmethod
    @mark.unit_test
    def test_start_nok():
        """ Test error cases for the start method of PipelineRunner """
        # 1 - test starting a pipeline with no subject settings
        runner = PipelineRunner('2T6S')
        runner.pipeline.directories.results_dir = '/home/bclenet/dev/narps_open_pipelines/run/'
        runner.pipeline.directories.set_working_dir_with_team_id('2T6S')
        runner.pipeline.directories.set_output_dir_with_team_id('2T6S')

        with raises(Exception):
            runner.start()

        # 2 - test starting a pipeline with wrong worflow type
        runner = PipelineRunner('2T6S')
        runner._pipeline = MockupWrongPipeline() # hack the runner by setting a test Pipeline

        with raises(AttributeError):
            runner.start()

        # 2b - test starting a pipeline with wrong worflow type
        runner = PipelineRunner('2T6S')
        runner._pipeline = MockupWrongPipeline2() # hack the runner by setting a test Pipeline

        with raises(AttributeError):
            runner.start()

        # 2b - test starting a pipeline with wrong options (fist_level_only + group_level_only)
        runner = PipelineRunner('2T6S')
        runner._pipeline = MockupPipeline() # hack the runner by setting a test Pipeline

        with raises(AttributeError):
            runner.start(True, True)

    @staticmethod
    @mark.unit_test
    def test_start_ok():
        """ Test normal usecases of PipelineRunner """
        # 1 - test starting a pipeline where everything ok
        runner = PipelineRunner('2T6S')
        runner._pipeline = MockupPipeline() # hack the runner by setting a test Pipeline
        runner.start()

        # 1 - read results of the pipeline
        with open(
            join(Configuration()['directories']['test_runs'], 'test_runner.txt'),
            'r', encoding = 'utf-8') as file:
            for workflow in [
                'TestPipelineRunner_preprocessing_workflow',
                'TestPipelineRunner_run_level_workflow',
                'TestPipelineRunner_subject_level_workflow',
                'TestPipelineRunner_group_level_workflow']:
                assert file.readline() == 'MockupPipeline : '+workflow+' node_1\n'
                assert file.readline() == 'MockupPipeline : '+workflow+' node_2\n'

        # 2 - test starting a pipeline partly (group_level_only)
        runner = PipelineRunner('2T6S')
        runner._pipeline = MockupPipeline() # hack the runner by setting a test Pipeline
        runner.start(False, True)

        # 2 - read results of the pipeline
        with open(
            join(Configuration()['directories']['test_runs'], 'test_runner.txt'),
            'r', encoding = 'utf-8') as file:
            assert file.readline() == \
                'MockupPipeline : TestPipelineRunner_group_level_workflow node_1\n'
            assert file.readline() == \
                'MockupPipeline : TestPipelineRunner_group_level_workflow node_2\n'

        # 3 - test starting a pipeline partly (first_level_only)
        runner = PipelineRunner('2T6S')
        runner._pipeline = MockupPipeline() # hack the runner by setting a test Pipeline
        runner.start(True, False)

        # 3 - read results of the pipeline
        with open(
            join(Configuration()['directories']['test_runs'], 'test_runner.txt'),
            'r', encoding = 'utf-8') as file:
            for workflow in [
                'TestPipelineRunner_preprocessing_workflow',
                'TestPipelineRunner_run_level_workflow',
                'TestPipelineRunner_subject_level_workflow']:
                assert file.readline() == 'MockupPipeline : '+workflow+' node_1\n'
                assert file.readline() == 'MockupPipeline : '+workflow+' node_2\n'
