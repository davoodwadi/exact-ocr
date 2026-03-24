# TODO: Make `pdf_to_markdown.sh` Interactive

## Current State
The script (`pdf_to_markdown.sh`) does a flat glob (`*.pdf`) in the current directory and batch-processes every PDF it finds, passing them all at once to `transcribe_qwen.py`.

---

## Steps to Implement

### 1. Recursive PDF Discovery
- Replace `ls -1 *.pdf` with a `find` command that walks subdirectories:
  ```bash
  mapfile -d '' pdf_files < <(find . -iname "*.pdf" -print0 | sort -z)
  ```
- Store results in an array so we can index into them later.

### 2. Filter Out Already-Converted PDFs
- For each discovered PDF, derive the expected markdown output path (same base name, `.md` extension, same directory or a known output dir).
- Remove PDFs that already have a corresponding `.md` file from the candidate list before showing the menu.
- Example check:
  ```bash
  md_path="${pdf_path%.pdf}.md"
  [[ -f "$md_path" ]] && continue   # skip — already converted
  ```

### 3. Interactive Selection Menu via `fzf`
Use [`fzf`](https://github.com/junegunn/fzf) for arrow-key navigation and multi-select — no numbered menu needed.

```bash
# Require fzf
if ! command -v fzf &>/dev/null; then
    echo "Error: fzf is required. Install with: sudo apt install fzf  OR  brew install fzf"
    exit 1
fi

mapfile -t selected < <(
    printf '%s\n' "${unprocessed[@]}" \
    | fzf --multi \
          --prompt="Select PDFs (Tab=toggle, Enter=confirm, Ctrl-A=all): " \
          --header="Arrow keys to navigate | Tab to select/deselect | Enter to confirm" \
          --preview='echo "PDF: {}"'
)
```

Key `fzf` flags:
- `--multi` — enables multi-select with Tab / Shift-Tab.
- `Ctrl-A` selects all; `Ctrl-D` / `Escape` quits.
- No extra "process all" branch needed — the user just hits `Ctrl-A` then `Enter`.
- Falls back gracefully: if the user presses Escape, `selected` is empty and the script exits cleanly.

### 4. Process the Queue
- After selection, iterate over the chosen files and call the Python script once per file (or batch them together — confirm which mode `transcribe_qwen.py` prefers):
  ```bash
  for pdf in "${selected[@]}"; do
      echo "Processing: $pdf"
      "$PYTHON_EXE" "$SCRIPT_PATH" "$pdf" "$@"
  done
  ```
- Print a summary when all selected files are done.

### 5. Edge Cases & UX Polish
- If `find` returns zero unprocessed PDFs, print a friendly message and exit.
- Confirm the selection with the user before starting (optional `--yes` flag to skip confirmation).
- Show progress as each file is processed (e.g., `[1/3] Processing foo.pdf ...`).
- Preserve pass-through of extra CLI flags (the existing `"$@"` pattern) so options like `-w` or `-e` still work.

---

## Open Questions
- [ ] Should the markdown output live next to the PDF or in a centralised output directory?
- [ ] Should already-converted files be listable via a `--show-done` flag?
- [ ] Single-pass batch call vs. one call per PDF — check `transcribe_qwen.py` argument handling.
