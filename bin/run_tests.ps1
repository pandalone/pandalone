## Run Python tests #under Windows

param([string]$test = 'auto')

$myname = $MyInvocation.MyCommand | Select -Expand Name
$tests = 'AUTO', 'all', 'cover', 'code', 'site', 'readme'
$tests_msg = $tests -join ' | '
$help = @"
Usage: 
    $myname [test_name]

Where: 
    'test_name': ( $tests_msg )

When 'auto' or unspecified, invokes what's proper for python-version.

"@


function test_readme() {
    echo "+++ Checking README for PyPy...."
    $rst2html = &where.exe rst2html.py
    python setup.py --long-description | &python "$rst2html" --halt=warning > $null
}

function test_site() {
    echo "+++ Checking site...."
    python setup.py build_sphinx
}

function test_code() {
    echo "+++ Checking only TCs...."
    nosetests "--where=tests" 
}
function test_cover() {
    echo "+++ Checking all TCs, DTs & coverage...."
    
    nosetests "--debug=" "--ignore-files=dodo.*" `
    "--tests=tests,README.rst" `
    "--with-doctest" "--doctest-extension=.rst" `
    "--exclude=dodo.py" `
    "--doctest-options=+NORMALIZE_WHITESPACE,+ELLIPSIS,+REPORT_UDIFF" `
    "--with-coverage" `
    "--cover-package=pandalone.components,pandalone.mappings,pandalone.pndlcmd,pandalone.utils,pandalone.xlsutils" `
    "--cover-min-percentage=60"
}

function test_all() {
    test_readme
    test_site
    test_cover
}

function decide_test() {
    $pyver = python --version
    if ($pyver) {
        return "all"
    } else {
        return "code"
    }
}

function main($test="auto") {
    if ($test -eq 'auto') {
        $test = decide_test
    }
    echo "---Runing $test..."
    invoke-expression "test_$test"
}



if ($test -eq "help" -or -not ($test -in $tests)) {
    echo $help
} else {
    main($test)
}
