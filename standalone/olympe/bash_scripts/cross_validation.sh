#!/bin/bash

cross_validation() {
    DATASET=$1
    CROSSVAL=$2

    # Fake array to ensure exactly one run. Not use for GSC and audioset.
    mvar="-t 1 -v 2"

    # prepare cross validation parameters
    # ---- default, no crossvalidation
    if [ "$DATASET" = "ubs8k" ]; then
        mvar="-t 1 2 3 4 5 6 7 8 9 -v 10"
    elif [ "$DATASET" = "esc10" ] || [ "$DATASET" = "esc50" ]; then
        mvar="-t 1 2 3 4 -v 5"
    fi

    # if crossvalidation is activated
    if [ $CROSSVAL -eq 1 ]; then
        if [ "$DATASET" = "ubs8k" ]; then
            mvar=$(python -c "import DCT.util.utils as u; u.create_bash_crossvalidation(10)")

        elif [ "$DATASET" = "esc10" ] || [ "$DATASET" = "esc50" ]; then
            mvar=$(python -c "import DCT.util.utils as u; u.create_bash_crossvalidation(5)")
        fi
    fi

    echo $mvar
}
