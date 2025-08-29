#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

echo "Launching all .sh files in: $SCRIPT_DIR"
echo "Excluding: $SCRIPT_NAME"
echo "----------------------------------------"

# Find all .sh files in the same directory, excluding this script
for script in "$SCRIPT_DIR"/*.sh; do
    # Check if the file exists and is not this script
    if [[ -f "$script" && "$(basename "$script")" != "$SCRIPT_NAME" ]]; then
        echo "Executing: $(basename "$script")"
        
        # Make the script executable if it isn't already
        chmod +x "$script"
        
        # Execute the script
        "$script"
        
        # Check if the script executed successfully
        if [[ $? -eq 0 ]]; then
            echo "✓ Successfully executed: $(basename "$script")"
        else
            echo "✗ Failed to execute: $(basename "$script")"
        fi
        
        echo "----------------------------------------"
    fi
done

echo "All scripts have been processed."
