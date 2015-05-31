#!/bin/bash
#-*- coding: utf-8 -*-
#
# Copyright 2015 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

## Checks that site's Sphinx docs do not produce WARNINGS!

set +x

my_dir=`dirname "$0"`
cd $my_dir/..

python setup.py build_sphinx 2>&1 | grep -v 'image' | grep WARNING && exit -1
echo OK
