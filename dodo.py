#! doit -f
"""
Usage:

- Type ``doit list`` to see list of available tasks to run.
- Type ``doit -v 2 <task>`` to run a task.
"""

from doit.tools import create_folder
import glob
import os

from pandalone import utils
from doit.exceptions import InvalidTask


mydir = os.path.dirname(__file__)
DOIT_CONFIG = {
    'minversion': '0.24.0',
    'default_tasks': [],
}

SAMPLES_FOLDER = os.path.join(mydir, 'pandalone', 'projects/')
SAMPLE_NAMES = [os.path.splitext(f)[0] for f in glob.glob1(SAMPLES_FOLDER, "*")]
SAMPLE_EXT = '.pndl'

opt_sample = {
    'help': 'name of the sample project;one of: %s' % SAMPLE_NAMES,
    'name': 'sample',
    'long': 'sample',
    'type': str,
    'default': 'simple_rpw',
}


def task_createsam():
    """doit createsam [target_dir]: Create new sample pandalone project as `target_dir`. """

    def copy_sample(sample, target_dir=None):
        if sample not in SAMPLE_NAMES:
            msg = '--sample: Unknown sample(%s)! Must be one of: %s'
            raise ValueError(msg % (sample, SAMPLE_NAMES))
        sample = utils.ensure_file_ext(sample, SAMPLE_EXT)
        if not target_dir:
            target_dir = '%s%s' % (sample, SAMPLE_EXT)
        else:
            if len(target_dir) > 1:
                msg = 'Too many `target_dir`s! \n\nUsage: \n  %s'
                raise InvalidTask(msg % task_createsam.__doc__)
            target_dir = target_dir[0]
        target_dir = utils.make_unique_filename(target_dir)
        print('Creating {sample} --> {target_dir}'.format(sample=sample, target_dir=target_dir))

    return {
        'actions': [copy_sample],
        'params': [opt_sample, ],
        'pos_arg': 'target_dir',
    }

