#!/bin/bash
# Called by "git commit" with no arguments.  The hook should
# exit with non-zero status after issuing an appropriate message if
# it wants to stop the commit.
STATUS=0

# Insert filemarkers to satisfy Apache license
PYFILES="$(find . -type f -name '*.py')"
for FILENAME in $PYFILES; do
    # if file doesn't already contain the file marker
    if ! grep -q "Modified by Joshua Shields" $FILENAME; then
        # mark file and set status to stop the commit
        printf "\n# Modified by Joshua Shields\n" >> $FILENAME
        echo "Marked $FILENAME, will stop the commit so change can be staged"
        STATUS=1
    fi
done

echo "Pre-commit hook exiting with status $STATUS"
exit $STATUS
