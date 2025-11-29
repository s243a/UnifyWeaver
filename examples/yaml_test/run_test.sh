#!/bin/bash
set -e

# Ensure we are in the project root
cd "$(dirname "$0")/../../.."

echo "Generating Bash script from YAML source..."
swipl -g "test_yaml_users, halt" -t halt examples/yaml_test/test_compile.pl > examples/yaml_test/generated_yaml_script.sh

echo "Running generated script..."
chmod +x examples/yaml_test/generated_yaml_script.sh
source examples/yaml_test/generated_yaml_script.sh

echo "--- Output of get_users ---"
get_users
echo "---------------------------"

echo "Done."
