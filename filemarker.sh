#!/bin/bash

PYFILES="$(ls **/*.py)"
#echo $PYFILES
for FILENAME in $PYFILES; do
    #echo $FILENAME
    echo "# Modified by Joshua Shields" >> $FILENAME
done

