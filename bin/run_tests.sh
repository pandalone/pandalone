#!/bin/bash
#-*- coding: utf-8 -*-
#
# Copyright 2013-2015 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

## From http://stackoverflow.com/a/25515370/548792
yell() { echo "$0: $*" >&2; }
die() { yell "$*"; exit 111; }

## Always invokes  code-tests, but skip doctests and coverage on Python-2.


my_dir=`dirname "$0"`
cd $my_dir/..

declare -A fails

echo "+++ Checking README for PyPI repo...."
./bin/check_readme.sh || fails['README']=$?

echo "+++ Checking SITE...."
./bin/check_site.sh || fails['site']=$?

echo "+++  Checking ARCHIVES..."
out="$( python setup.py sdist bdist_wheel  2>&1 )"
if [ $? -ne 0 ]; then
	fails['archives']=1
    yell "$out ARCHIVES failed!"
else
	echo "OK"
fi

if  python -c 'import sys;  exit(not sys.version_info >= (3,4))'; then
    echo "+++ Checking all TCs, DTs & Coverage....";
	out="$( python setup.py test_all  2>&1 )"
	if [ $? -ne 0 ]; then
		fails['code_full']=1
	    yell "$out ALL_TCs failed!"
	else
		echo "OK"
	fi
else
    echo "+++ Checking only TCs....";
	out="$( python setup.py test_code 2>&1 )"
	if [ $? -ne 0 ]; then
	    yell "$out CODE_TCs failed!"
	    fails['code_only']=1
	else
		echo "OK"
	fi
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
