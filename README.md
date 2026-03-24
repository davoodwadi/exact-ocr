# Exact OCR

An AI-powered OCR and document extraction tool that uses Vision-Language Models (specifically Qwen) via vLLM to transcribe PDFs and images into structured Markdown. The tool is capable of intelligently extracting text, tables, and embedding relevant figures directly into the Markdown file.

Additionally, this repository supports generating beautiful, styled PDFs and EPUBs inspired by the manuscript aesthetic.

## Features

- **Intelligent Transcription**: Leverages **Qwen3.5-9B** (via vLLM) to convert entire PDF pages or individual images into clean, structured Markdown, accurately converting tables and preserving text structure.
- **Image Extraction**: Automatically identifies and extracts figures, charts, and diagrams from PDFs, omitting decorative elements and logos, and embeds them within the Markdown output.
- **Smart Metadata Parsing**: Extracts Title, Author, Date, and Keywords, saving them as YAML frontmatter in your Markdown documents.
- **Alan Wake Manuscript Styling**: Built-in support to convert the transcribed Markdown into a beautifully styled PDF or EPUB formatted like the Alan Wake 2 manuscript pages using HTML/CSS rules.
- **High Performance**: Runs batch inference on multiple pages simultaneously using vLLM for fast processing.
- **Raw Text Bypass Mode**: Fast, LLM-bypassed extraction of raw text straight from the PDF for simple documents.
- **Batch Processing**: A handy shell script (`pdf_to_markdown.sh`) to effortlessly transcribe all PDFs in a directory.

## Prerequisites

To run this tool, you'll need:
- **Python 3.8+**
- A machine capable of running **vLLM** (Linux or WSL is highly recommended, along with a compatible NVIDIA GPU with sufficient VRAM for a 9B model).
- **Pandoc** and **WeasyPrint** (Required *only* if you want to generate Alan Wake styled PDFs and EPUBs).

### Install System Dependencies

If you plan on generating PDFs or EPUBs, install Pandoc and WeasyPrint on your system:

**Ubuntu / Debian:**
```bash
sudo apt-get update
sudo apt-get install pandoc weasyprint
```

**macOS (via Homebrew):**
```bash
brew install pandoc weasyprint
```

### Install Python Dependencies

Clone this repository and install the necessary Python packages:

```bash
git clone git@github.com:davoodwadi/exact-ocr.git
cd exact-ocr
pip install -r requirements.txt
```

## Usage

The core transcription engine is `ocr/transcribe_qwen.py`. 

### Basic Transcription
To extract content from a single PDF or image and convert it to Markdown:

```bash
python ocr/transcribe_qwen.py path/to/your/document.pdf
```
By default, this will spin up a local vLLM OpenAI-compatible server on a free port, process the document page-by-page, smartly stitch them together, and output a `document.md` in the same directory as the input file. Extracted images are saved in an `extracted_images_document/` directory.

### Command Line Arguments

You can heavily customize the behavior of the script using the following flags:

| Flag | Long Flag | Description |
| :--- | :--- | :--- |
| `-e` | `--epub` | Generate an Alan Wake-styled EPUB file alongside the Markdown. |
| `-w` | `--wake` | Generate an Alan Wake-styled PDF file alongside the Markdown. |
| `-m` | `--max-pages` | Set a limit on the number of pages to process (e.g., `-m 10` for the first 10 pages). |
| `-f` | `--force` | Force re-transcription even if a `.md` output file already exists. |
| `-P` | `--python-api` | Use the vLLM Python API directly instead of spinning up an OpenAI server. |
| `-t` | `--text-only` | Bypass the LLM and extract raw text from the PDF directly (faster, but less intelligent formatting). |

### Examples

**Transcribe a PDF and create an Alan Wake styled EPUB and PDF:**
```bash
python ocr/transcribe_qwen.py document.pdf -e True -w True
```

**Force re-transcribe only the first 5 pages of a document using the Python API:**
```bash
python ocr/transcribe_qwen.py document.pdf -m 5 -f -P
```

**Bypass the LLM and quickly extract raw text:**
```bash
python ocr/transcribe_qwen.py document.pdf -t
```

---

## Batch Processing

If you have a directory filled with PDFs, you can use the included bash script `pdf_to_markdown.sh` to batch-process all of them in a single transcription run.

The script automatically searches the current directory for any `.pdf` files and passes them to `transcribe_qwen.py`.

```bash
# Run batch transcription on all PDFs in the current folder
./pdf_to_markdown.sh

# You can pass any of the Python script arguments to the bash script!
# Example: Batch process and generate EPUBs for all PDFs
./pdf_to_markdown.sh -e True
```

*Note: You may need to make the script executable first by running `chmod +x pdf_to_markdown.sh`.*
