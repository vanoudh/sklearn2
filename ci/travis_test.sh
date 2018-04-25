#!/usr/bin/env bash

set -e


if [[ "$COVERAGE" == "true" ]]; then
    nosetests -s -v --with-coverage
else
	nosetests -s -v
fi

