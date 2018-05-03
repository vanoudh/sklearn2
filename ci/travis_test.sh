#!/usr/bin/env bash

set -e

conda install pytest
pytest -v
