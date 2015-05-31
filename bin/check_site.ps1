#!/bin/bash
#-*- coding: utf-8 -*-
#
# Copyright 2015 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

## Checks that site's Sphinx docs do not produce WARNINGS!

$mydir=Split-Path $script:MyInvocation.MyCommand.Path
cd $mydir/..


$site_out = &python setup.py build_sphinx 2>&1 |
        select-string  -not -pattern 'image'

if ($lastexitcode -ne 0) { 
    throw "Site-failed! `n`n`n`n$site_out"
}
if ($site_out | select-string -CaseSensitive -pattern  "WARNING") {
    throw "WARNINGs detected! `n`n`n`n$site_out"
}
