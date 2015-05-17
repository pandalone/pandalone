#!/usr/bin/env python

""" Git Versioning Script

Will transform stdin to expand some keywords with git version/author/date information.

Specify --clean to remove this information before commit.

Setup:

1. [DONE] Setup sources::

    git add bin/gitstamping.py
    echo '*.py filter=gitstamping' >> .gitattributes
 

2. [DONE] Add a within files this statements::

    __commit__ = ""


3. Setup your working-dir::

    git config filter.gitstamping.smudge 'python bin/gitstamping.py'
    git config filter.gitstamping.clean  'python bin/gitstamping.py --clean'


From https://gist.github.com/pkrusche/7369262
"""

import re
import subprocess
import sys


def filter_or_smudge(clean):
    rexp1 = re.compile(r'__commit__(\s*)=(\s*)".*')
    if clean:
        for line in sys.stdin:
            line = re.sub(rexp1, r'__commit__\1=\2""', line)
            sys.stdout.write(re.sub(rexp1, r'__commit__\1=\2""', line))
    else:
        git_id = subprocess.check_output(['git', 'describe', '--always'])
        git_id = git_id.decode(encoding='utf_8')
        git_id = re.sub(r'[\n\r\t"\']', "", git_id)
        for line in sys.stdin:
            line = re.sub(rexp1, r'__commit__\1=\2"%s"' % git_id, line)
            sys.stdout.write(line)


def main(args):
    clean = len(sys.argv) > 1 and sys.argv[1] == '--clean'
    filter_or_smudge(clean)

if __name__ == "__main__":
    main(sys.argv)
