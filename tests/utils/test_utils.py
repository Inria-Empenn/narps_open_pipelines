#!/usr/bin/python
# coding: utf-8

""" Tests of the 'narps_open.utils' module.

Launch this test with PyTest

Usage:
======
    pytest -q test_utils.py
    pytest -q test_utils.py -k <selected_test>
"""
from os.path import join

from pytest import mark

from narps_open.utils.configuration import Configuration
from narps_open.utils import show_download_progress, hash_image, hash_dir_images

class TestUtils:
    """ A class that contains all the unit tests for the utils module."""

    @staticmethod
    @mark.unit_test
    def test_show_download_progress(capfd): # using pytest's capfd fixture to get stdout
        """ Test the show_download_progress function """

        show_download_progress(25,1,100)
        captured = capfd.readouterr()
        assert captured.out == 'Downloading 25 %\r'

        show_download_progress(26,2,200)
        captured = capfd.readouterr()
        assert captured.out == 'Downloading 26 %\r'

        show_download_progress(25,50,-1)
        captured = capfd.readouterr()
        assert captured.out == 'Downloading ⣽\r'

    @staticmethod
    @mark.unit_test
    def test_hash_image():
        """ Test the hash_image function """

        # Get test_data for hash
        test_image_path = join(
            Configuration()['directories']['test_data'],
            'utils', 'hash', 'hypo1_unthresh.nii.gz'
            )

        value = '755cee10777bc3b3a9707eb20a46793d282fedc07a52d6c4a9866e465fd6ccb3'
        assert hash_image(test_image_path) == value

    @staticmethod
    @mark.unit_test
    def test_hash_dir_images():
        """ Test the hash_dir_images function """

        # Get test_data for hash
        test_path = join(
            Configuration()['directories']['test_data'],
            'utils', 'hash'
            )

        value = '4242d5eb8d4c0dc70adcec11154ab029c3b1dcdfb777c5dff4ffcff1f1ff6acb'
        assert hash_dir_images(test_path) == value
