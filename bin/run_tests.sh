#!/bin/bash
#-*- coding: utf-8 -*-
#
# Copyright 2013-2015 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl


## Always invokes  code-tests, but skip doctests and coverage on Python-2.


my_dir=`dirname "$0"`
cd $my_dir/..

declare -A fails

echo "+++ Checking README for PyPy...."
./bin/check_readme.sh || fails['README']=$?

echo "+++ Checking site...."
( python setup.py build_sphinx 2>&1 | grep -v 'image' | grep WARNING ) || fails['site']=$?

echo "+++  Checking archives for PyPI repo..."
python setup.py sdist bdist_wheel || fails['archives']=$?

if  python -c 'import sys; print(sys.version_info[0])'| grep -q '3'; then
    echo "+++ Checking all TCs, DTs & Coverage....";
    python setup.py test_all || fails['all']=$?
else
    echo "+++ Checking only TCs....";
    python setup.py test_code || fails['code']=$?   
fi

## Raise any errors.
#
if [ ${#fails[@]} -gt 0 ]; then
    echo -n "TEST FAILURES:  "
    for fail in "${!fails[@]}"; do
        echo -n "${fail}(${fails[$fail]})"
    done
    echo

    exit 1
fi
