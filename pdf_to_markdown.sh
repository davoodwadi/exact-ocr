#!/bin/bash

# Path to the virtual environment Python
PYTHON_EXE="/home/dw/github/pdf-to-markdown/.venv/bin/python"

# Path to the transcription script
SCRIPT_PATH="/home/dw/github/pdf-to-markdown/ocr/transcribe_qwen.py"

# ---------------------------------------------------------------------------
# Parse script-level flags (consume before passing remainder to Python script)
# ---------------------------------------------------------------------------
YES=0
SHOW_DONE=0
PASSTHROUGH=()

for arg in "$@"; do
    case "$arg" in
        --yes|-y)   YES=1 ;;
        --show-done) SHOW_DONE=1 ;;
        *)          PASSTHROUGH+=("$arg") ;;
    esac
done

# ---------------------------------------------------------------------------
# Require fzf
# ---------------------------------------------------------------------------
if ! command -v fzf &>/dev/null; then
    echo "Error: fzf is required."
    echo "  Install: sudo apt install fzf   OR   brew install fzf"
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Recursive PDF discovery
# ---------------------------------------------------------------------------
mapfile -d '' all_pdfs < <(find . \( -type d -name '.*' ! -name . \) -prune -o -iname "*.pdf" -print0 | sort -z)

if [[ ${#all_pdfs[@]} -eq 0 ]]; then
    echo "No PDF files found under $(pwd)."
    exit 0
fi

# ---------------------------------------------------------------------------
# 2. Partition: unprocessed vs already-converted
# ---------------------------------------------------------------------------
unprocessed=()
already_done=()

for pdf in "${all_pdfs[@]}"; do
    md_path="${pdf%.pdf}.md"
    # Also check uppercase extension
    md_path_lower="${pdf%.PDF}.md"
    if [[ -f "$md_path" || -f "$md_path_lower" ]]; then
        already_done+=("$pdf")
    else
        unprocessed+=("$pdf")
    fi
done

# Optionally show already-converted files
if [[ $SHOW_DONE -eq 1 && ${#already_done[@]} -gt 0 ]]; then
    echo "Already converted (${#already_done[@]}):"
    printf '  %s\n' "${already_done[@]}"
    echo ""
fi

if [[ ${#unprocessed[@]} -eq 0 ]]; then
    echo "All discovered PDFs have already been converted."
    echo "Use --show-done to list them, or pass -f/--force to the Python script to re-process."
    exit 0
fi

# ---------------------------------------------------------------------------
# 3. Interactive selection via fzf
# ---------------------------------------------------------------------------
mapfile -t selected < <(
    printf '%s\n' "${unprocessed[@]}" \
    | fzf --multi \
          --prompt="Select PDFs to convert: " \
          --header="TAB = toggle | Ctrl-A = select all | Ctrl-D = deselect all | Enter = confirm | Esc = cancel" \
          --bind='ctrl-a:select-all,ctrl-d:deselect-all' \
          --marker='✓ ' \
          --pointer='>' \
          --color='marker:green,hl:green,hl+:green,fg+:white,bg+:#2a2a2a,pointer:green' \
          --preview='pdfinfo {} 2>/dev/null | grep -E "Title|Author|Pages|File size" || echo "(pdfinfo not available)"'
)

if [[ ${#selected[@]} -eq 0 ]]; then
    echo "No files selected. Exiting."
    exit 0
fi

# ---------------------------------------------------------------------------
# 4. Confirm before processing (skip with --yes)
# ---------------------------------------------------------------------------
if [[ $YES -eq 0 ]]; then
    echo ""
    echo "Selected for conversion (${#selected[@]}):"
    printf '  %s\n' "${selected[@]}"
    echo ""
    read -r -p "Proceed? [y/N] " confirm
    case "$confirm" in
        [yY][eE][sS]|[yY]) ;;
        *)
            echo "Aborted."
            exit 0
            ;;
    esac
fi

# ---------------------------------------------------------------------------
# 5. Process selection in a single batch invocation
# ---------------------------------------------------------------------------
total=${#selected[@]}

echo ""
echo "=================================================="
echo "Processing $total file(s)"
echo "=================================================="

echo ""
printf '  %s\n' "${selected[@]}"
echo ""
"$PYTHON_EXE" "$SCRIPT_PATH" "${selected[@]}" "${PASSTHROUGH[@]}"
status=$?

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=================================================="
if [[ $status -eq 0 ]]; then
    echo "Done. Submitted $total file(s) in a single transcription run."
else
    echo "Batch transcription exited with status $status."
    echo "  Check output above for details."
fi
echo "=================================================="

exit $status
