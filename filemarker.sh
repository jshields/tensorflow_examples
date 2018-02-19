#!/bin/bash

# Insert filemarkers to satisfy Apache license
PYFILES="$(find . -type f -name '*.py')"
for FILENAME in $PYFILES; do
    # if file doesn't already contain the file marker
    if ! grep -q "Modified by Joshua Shields" $FILENAME; then
        echo "Marked $FILENAME"
        printf "\n# Modified by Joshua Shields\n" >> $FILENAME
    fi
done

echo "Pre-commit hook executed"
exit 1
