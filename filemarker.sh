#!/bin/bash

STATUS=0

# Insert filemarkers to satisfy Apache license
PYFILES="$(find . -type f -name '*.py')"
for FILENAME in $PYFILES; do
    # if file doesn't already contain the file marker
    if ! grep -q "Modified by Joshua Shields" $FILENAME; then
        # mark file and set status to stop the commit
        printf "\n# Modified by Joshua Shields\n" >> $FILENAME
        echo "Marked $FILENAME"
        STATUS=1
    fi
done

echo "Pre-commit hook exiting with status $STATUS"
exit $STATUS
