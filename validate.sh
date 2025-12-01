#!/bin/bash
# Wrapper script to run data validation from project root

cd "$(dirname "$0")/scripts" && python validate_data.py "$@"
