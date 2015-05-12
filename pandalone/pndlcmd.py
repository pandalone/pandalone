#! doit -f
# -*- coding: UTF-8 -*-
#
# Copyright 2015 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Usage:

- Type ``doit list`` to see list of available tasks to run.
- Type ``doit -v 2 <task>`` to run a task.
"""
## Ideas also from:
#  https://realpython.com/blog/python/scaffold-a-flask-project/?utm_source=Python+Weekly+Newsletter&utm_campaign=dcba979511-Python_Weekly_Issue_189_April_30_2015&utm_medium=email&utm_term=0_9e26887fc5-dcba979511-312746653 
from doit.exceptions import InvalidTask
from doit.tools import create_folder
import glob
import os
from pandalone import utils
import shutil


mydir = os.path.dirname(__file__)
DOIT_CONFIG = {
    'minversion': '0.24.0',
    'default_tasks': [],
}

SAMPLES_FOLDER = os.path.join(mydir, 'sample_projects/')
SAMPLE_NAMES = [os.path.splitext(f)[0]
                for f in glob.glob1(SAMPLES_FOLDER, "*")]
SAMPLE_EXT = '.pndl'

opt_sample = {
    'help': 'name of the sample project;one of: %s' % SAMPLE_NAMES,
    'name': 'sample',
    'long': 'sample',
    'type': str,
    'default': 'simple_rpw',
}


def task_makesam():
    """doit makesam [target_dir]: Create new sample pandalone project as `target_dir`. """

    def copy_sample(sample, target_dir=None):
        if os.path.splitext(sample)[0] not in SAMPLE_NAMES:
            msg = '--sample: Unknown sample(%s)! Must be one of: %s'
            raise ValueError(msg % (sample, SAMPLE_NAMES))
        sample_dir = utils.ensure_file_ext(sample, SAMPLE_EXT)
        if not target_dir:
            target_dir = sample_dir
        else:
            if len(target_dir) > 1:
                msg = 'Too many `target_dir`s! \n\nUsage: \n  %s'
                raise InvalidTask(msg % task_makesam.__doc__)
            target_dir = target_dir[0]
        target_dir_unique = utils.make_unique_filename(target_dir)
        opening = ("Dir '%s' already exists! " % target_dir
                   if target_dir != target_dir_unique
                   else '')
        
        print('%sCopying %s --> %s' % (opening, sample_dir, target_dir_unique))
        
        srcdir = os.path.join(SAMPLES_FOLDER, sample_dir)
        shutil.copytree(srcdir, target_dir_unique)
        shutil.copy(os.path.join(mydir, '..', '.gitignore'), target_dir_unique)

    return {
        'actions': [copy_sample],
        'params': [opt_sample, ],
        'pos_arg': 'target_dir',
    }
