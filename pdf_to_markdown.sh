#!/bin/bash

# Path to the virtual environment Python
PYTHON_EXE="/home/dw/github/wake-manuscript/.venv/bin/python"

# Path to the transcription script
SCRIPT_PATH="/home/dw/github/wake-manuscript/ocr/transcribe_qwen.py"

# Check if there are any PDF files in the current directory
count=$(ls -1 *.pdf 2>/dev/null | wc -l)

if [ "$count" -eq 0 ]; then
    echo "No PDF files found in the current directory."
    exit 0
fi

echo "Found $count PDF file(s). Processing..."

echo "=================================================="
echo "Batch Processing all PDFs with $@"
echo "=================================================="
# Pass all PDF files and any extra arguments to the script (e.g., -w, -e)
"$PYTHON_EXE" "$SCRIPT_PATH" *.pdf "$@"

echo "All PDFs processed."
