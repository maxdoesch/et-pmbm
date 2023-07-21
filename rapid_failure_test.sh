#!/bin/bash

program_to_restart="/home/max/et_pmbm/build/test_detection_group"  # Replace this with the actual path to your program

while true; do
    # Run the program and capture its exit code
    $program_to_restart
    exit_code=$?

    # Check the exit code
    if [ $exit_code -eq 0 ]; then
        echo "Program exited successfully."
    else
        echo "Program exited with an error (Exit code: $exit_code)."
	break
    fi
done

