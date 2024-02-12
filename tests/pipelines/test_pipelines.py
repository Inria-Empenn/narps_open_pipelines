#!/usr/bin/python
# coding: utf-8

""" Tests of the 'pipelines' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_pipelines.py
    pytest -q test_pipelines.py -k <selected_test>
"""

from pytest import raises, mark

from narps_open.pipelines import (
    PipelineDirectories,
    Pipeline,
    get_not_implemented_pipelines,
    get_implemented_pipelines
    )

class InheritingErrorPipeline(Pipeline):
    """ A class to test the inheritance of the Pipeline class """

class InheritingPipeline(Pipeline):
    """ A class to test the inheritance of the Pipeline class """

    def __init__(self):
        super().__init__()

        self._fwhm = 8.0

    def get_preprocessing(self):
        return 'a'

    def get_run_level_analysis(self):
        return 'r'

    def get_subject_level_analysis(self):
        return 'b'

    def get_group_level_analysis(self):
        return 'c'

    def get_hypotheses_outputs(self):
        return ['h1', 'h2']

class TestPipelineDirectories:
    """ A class that contains all the unit tests for the PipelineDirectories class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a PipelineDirectories object """

        # 1 - create the object
        pipeline_dir = PipelineDirectories()

        # 2 - check one can access the properties of the class
        pipeline_dir.dataset_dir = 'test_d'
        assert pipeline_dir.dataset_dir == 'test_d'

        pipeline_dir.results_dir = 'test_r'
        assert pipeline_dir.results_dir == 'test_r'

        pipeline_dir.working_dir = 'test_w'
        assert pipeline_dir.working_dir == 'test_w'

        pipeline_dir.output_dir = 'test_o'
        assert pipeline_dir.output_dir == 'test_o'

    @staticmethod
    @mark.unit_test
    def test_alternative_setters():
        """ Test the alternatives setters of PipelineDirectories """

        # 1 - create the object
        pipeline_dir = PipelineDirectories()

        # 2 - check generation of the directories without setting results_dir
        with raises(AttributeError):
            pipeline_dir.set_working_dir_with_team_id('2T6S')
        with raises(AttributeError):
            pipeline_dir.set_output_dir_with_team_id('2T6S')

        # 3 - check generation of the directories after results_dir is set
        pipeline_dir.results_dir = 'pipeline_dir._results_dir'
        pipeline_dir.set_working_dir_with_team_id('2T6S')
        pipeline_dir.set_output_dir_with_team_id('2T6S')

        test_str = 'pipeline_dir._results_dir/NARPS-2T6S-reproduced/intermediate_results'
        assert pipeline_dir.working_dir == test_str
        assert pipeline_dir.output_dir == 'pipeline_dir._results_dir/NARPS-2T6S-reproduced'

class TestPipelines:
    """ A class that contains all the unit tests for the Pipeline class."""

    @staticmethod
    @mark.unit_test
    def test_create():
        """ Test the creation of a Pipeline object """

        # 1 - check that creating a Pipeline object is not possible (abstract class)
        with raises(TypeError):
            pipeline = Pipeline()

        # 2 - check the instantiation of an incomplete class inheriting from Pipeline
        with raises(TypeError):
            pipeline = InheritingErrorPipeline()

        # 3 - check the instantiation of a class inheriting from Pipeline
        pipeline = InheritingPipeline()

        # 4 - check accessing the attributes of Pipeline through an inheriting class
        assert isinstance(pipeline.directories, PipelineDirectories)

        assert pipeline.tr == 1.0
        with raises(AttributeError):
            pipeline.tr = 2.0

        assert pipeline.fwhm == 8.0
        pipeline.fwhm = 4.0
        assert pipeline.fwhm == 4.0

        assert pipeline.get_preprocessing() == 'a'
        assert pipeline.get_subject_level_analysis() == 'b'
        assert pipeline.get_group_level_analysis() == 'c'
        assert pipeline.get_hypotheses_outputs()[0] == 'h1'

class TestUtils:
    """ A class that contains all the unit tests for the utils in module pipelines."""

    @staticmethod
    @mark.unit_test
    def test_utils():
        """ Test the utils methods of PipelineRunner """
        # 1 - Get not implemented pipelines
        assert '1K0E' in get_not_implemented_pipelines()

        # 2 - Get implemented pipelines
        assert '2T6S' in get_implemented_pipelines()
