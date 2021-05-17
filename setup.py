#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import codecs
from setuptools import setup, find_packages


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

# https://github.com/pypa/setuptools_scm
use_scm = {"write_to": "napari_plugin_test/_version.py"}

setup(
    use_scm_version=use_scm,
	name='napari-test',
	author='Daniel Krentzel',
	author_email='dkrentzel@pm.me',
	license='MIT',
	url='https://github.com/krentzd/napari-test',
	description='Testing cookiecutter',
	long_description=read('README.md'),
	long_description_content_type='text/markdown',
	classifiers=[
		'Development Status :: 2 - Pre-Alpha',
    		'Intended Audience :: Developers',
    		'Framework :: napari',
    		'Topic :: Software Development :: Testing',
    		'Programming Language :: Python',
    		'Programming Language :: Python :: 3',
    		'Programming Language :: Python :: 3.7',
    		'Programming Language :: Python :: 3.8',
    		'Programming Language :: Python :: 3.9',
    		'Operating System :: OS Independent',
    		'License :: OSI Approved :: MIT License',
	],
	entry_points={'napari.plugin': 'napari-test = napari_plugin_test'},
)
