#!/bin/bash

# Usage:
# ./trigger.sh "-trigger <trigger> -os <os> -pv <pv>"
#
# To trigger the workflow with trigger <trigger>, and to run all
# components of that workflow which use operating system <os> and
# Python version <pv>.
#
# e.g. ./trigger.sh "-trigger test_deployed -os ubuntu-16.04 -pv 3.7"
#
# Wildcards (*) are accepted for <os> and <pv>
#
# e.g. ./trigger.sh "-trigger test_deployed -os ubuntu-16.04 -pv *"
#
# Note the quotation marks.
#
# The value of <trigger> is typically the name of the file. but actually corresponds to the value of trigger argument
# specified in the file.

if [[ $# -ne 1 ]]; then
    echo Please supply exactly one argument
    exit 1
fi
if [[ $1 == '-not-available-' ]]; then
    echo "Not available."
    exit 1
fi

read -p "Username: " CURLUSER
curl -u $CURLUSER --data "{\"event_type\": \"$1\"}" -H "Accept: application/vnd.github.everest-preview+json" https://api.github.com/repos/patrick-kidger/signatory/dispatches
