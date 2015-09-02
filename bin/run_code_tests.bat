IF "%PYTHON_VERSION%"=="3.4" (
    echo "+++ Checking all TCs, DTs & Coverage...."
    python setup.py test_all
) ELSE (
    echo "+++ Checking only TCs...."
    python setup.py test_code
)
