#!/usr/bin/python
# coding: utf-8

""" Tests of the 'pipelines' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_pipelines.py
    pytest -q test_pipelines.py -k <selected_test>
"""

from pytest import raises

from narps_open.pipelines import PipelineDirectories, Pipeline

class InheritingErrorPipeline(Pipeline):
    """ A class to test the inheritance of the Pipeline class """

class InheritingPipeline(Pipeline):
    """ A class to test the inheritance of the Pipeline class """

    def __init__(self):
        super().__init__()

        self.fwhm = 8.0

    def get_preprocessing(self):
        return 'a'

    def get_subject_level_analysis(self):
        return 'b'

    def get_group_level_analysis(self):
        return 'c'

class TestPipelineDirectories:
    """ A class that contains all the unit tests for the PipelineDirectories class."""

    @staticmethod
    def test_create():
        """ Test the creation of a PipelineDirectories object """

        # 1a - create the object
        pipeline_dir = PipelineDirectories()

        # 1b - 'hard writing' the class attributes for test purposes
        pipeline_dir._dataset_dir = 'pipeline_dir._dataset_dir'
        pipeline_dir._working_dir = 'pipeline_dir._working_dir'
        pipeline_dir._output_dir = 'pipeline_dir._output_dir'

        # 2 - check one can access the following attributes
        #    - dataset_dir (get only)
        #    - working_dir (get and set)
        #    - output_dir (get and set)
        assert pipeline_dir.dataset_dir == 'pipeline_dir._dataset_dir'
        with raises(AttributeError):
            pipeline_dir.dataset_dir = 'test_d'

        assert pipeline_dir.working_dir == 'pipeline_dir._working_dir'
        pipeline_dir.working_dir = 'test_w'
        assert pipeline_dir.working_dir == 'test_w'

        assert pipeline_dir.output_dir == 'pipeline_dir._output_dir'
        pipeline_dir.output_dir = 'test'
        assert pipeline_dir.output_dir == 'test'

    @staticmethod
    def test_setup():
        """ Test the setup method of PipelineDirectories """

        # 1a - create the object
        pipeline_dir = PipelineDirectories()

        # 1b - 'hard writing' the class attributes for test purposes
        pipeline_dir._results_dir = 'pipeline_dir._results_dir'

        # 2 - 
        pipeline_dir.setup('team_id_test')
        test_str = 'pipeline_dir._results_dir/NARPS-team_id_test-reproduced/intermediate_results'
        assert pipeline_dir.working_dir == test_str
        assert pipeline_dir.output_dir == 'pipeline_dir._results_dir/NARPS-team_id_test-reproduced'

class TestPipelines:
    """ A class that contains all the unit tests for the Pipeline class."""

    @staticmethod
    def test_create():
        """ Test the creation of a Pipeline object """

        # 1 - check that creating a Pipeline object is not possible (abstract class)
        with raises(TypeError):
            pipeline = Pipeline()

        # 2 - check the instanciation of an incomplete class inheriting from Pipeline
        with raises(TypeError):
            pipeline = InheritingErrorPipeline()

        # 3 - check the instanciation of a class inheriting from Pipeline
        pipeline = InheritingPipeline()

        # 4 - check accessing the attributes of Pipeline through an inheriting class
        assert type(pipeline.directories) == PipelineDirectories

        
        assert pipeline.tr == 1.0
        with raises(AttributeError):
            pipeline.tr = 2.0

        assert pipeline.fwhm == 8.0
        pipeline.fwhm = 4.0
        assert pipeline.fwhm == 4.0

        assert pipeline.get_preprocessing() == 'a'
        assert pipeline.get_subject_level_analysis() == 'b'
        assert pipeline.get_group_level_analysis() == 'c'
