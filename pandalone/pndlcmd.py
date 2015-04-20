#! doit -f
"""
Usage:

- Type ``doit list`` to see list of available tasks to run.
- Type ``doit -v 2 <task>`` to run a task.
"""

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
