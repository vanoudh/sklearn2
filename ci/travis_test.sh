#!/usr/bin/env bash

set -e

python --version
python -c "import pandas; print('pandas %s' % pandas.__version__)"
nosetests -v