#!/usr/bin/python
# coding: utf-8

""" Tests of the 'runner' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_runner.py
    pytest -q test_runner.py -k <selected_test>
"""

from pytest import raises, mark

from nipype import Node, Workflow
from nipype.interfaces.utility import Split, Merge

from narps_open.runner import PipelineRunner
from narps_open.pipelines import Pipeline
from narps_open.pipelines.team_2T6S import PipelineTeam2T6S

class MockupPipeline(Pipeline):
    """ A simple Pipeline class for test purposes """

    def __init__(self):
        super().__init__()

    def get_preprocessing(self):
        node_1 = Node(
            Split(),
            name = 'node_1'
            )
        node_1.inputs.inlist = [1, 2, 3] # the list to split
        node_1.inputs.splits = [1, 2] # the number of elements per output lists

        node_2 = Node(
            Merge(2),
            name = 'node_2'
            )

        workflow = Workflow(base_dir = 'run', name = 'TestPipelineRunner_preprocessing_workflow')
        workflow.add_nodes([node_1, node_2])
        workflow.connect(node_1, 'out1', node_2, 'in1')
        workflow.connect(node_1, 'out2', node_2, 'in2')

        return workflow

    def get_run_level_analysis(self):
        return None

    def get_subject_level_analysis(self):
        return None

    def get_group_level_analysis(self):
        return None

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

    @staticmethod
    @mark.unit_test
    def test_start_ok():
        """ Test normal usecases of PipelineRunner """
        # 1 - test starting a pipeline where everything ok
        runner = PipelineRunner('2T6S')
        runner._pipeline = MockupPipeline() # hack the runner by setting a test Pipeline
        runner.start()
