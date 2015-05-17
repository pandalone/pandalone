#!/bin/bash
#-*- coding: utf-8 -*-
#
# Copyright 2013-2015 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl


## Checks that README has no RsT-syntactic errors.
# Since it is used by `setup.py`'s `description` if it has any errors, 
# PyPi would fail parsing them, ending up with an ugly landing page,
# when uploaded.


my_dir=`dirname "$0"`
cd $my_dir/..

declare -A fails
py=""
rst="rst2html"
if [ ! -x "`which rst2html`" ]; then
    rst="`which rst2html.py`"
    py=python
    if [ -x "`which cygpath`" ]; then
        rst="`cygpath -w $rst`"
    else
        echo -e "Cannot find 'rst2html'! \n Sphinx installed? `pip show sphinx`" && exit 1
    fi
fi

export PYTHONPATH='$my_dir/..'
python setup.py --long-description | $py "$rst"  --halt=warning > /dev/null
