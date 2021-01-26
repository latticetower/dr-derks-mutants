#!/bin/bash
eval $(egrep -v '^#' .ENV | xargs) "$@"