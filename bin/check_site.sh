#!/bin/bash
#-*- coding: utf-8 -*-
#
# Copyright 2015 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

## Checks that site's Sphinx docs do not produce WARNINGS!

## From http://stackoverflow.com/a/25515370/548792
yell() { echo "$0: $*" >&2; }
die() { yell "$*"; exit 111; }

set +x ## Enable for debug

my_dir=`dirname "$0"`
cd $my_dir/..

## Build-site and capture output.
#
out="$( python setup.py build_sphinx 2>&1 )"
if [ $? -ne 0 ]; then
    die "$out SPHINX failed!"
fi

## Check for warnings.
#
warns="$( echo "$out" |
    grep -v 'image' |
    grep -v ' WARNING: more than one target found for cross-reference' |
    egrep 'WARNING|ERROR' )"
if [ -n "$warns" ]; then
    die "$warns"
fi

echo OK
