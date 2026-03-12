# Alan Wake 2 Manuscript Project Instructions

## Project Context

This workflow takes a pdf for an academic article and turn it to manuscript pages from the game _Alan Wake 2_. It uses CSS to simulate a physical typewriter page with fonts and margins.

## Tech Stack

1. Use google genai api to extract text from pdf and create a markdown version of the pdf file. use the codegen_instructions.md in the root directory for the latest python version of the google genai api
2. convert the markdown to pdf in Alan Wake 2 style using pandoc:
   `pandoc input.md -o manuscript.pdf --css wake-style.css --pdf-engine=weasyprint -V lang=en-US`

## Style Guidelines

### 1. Typography and Base Layout

- **Font**: Prioritize **"American Typewriter"**, **"Courier Prime"**, or **"Courier New"**.

### 2. The Physical Paper

- **Dimensions**: 8.5x11 inch aspect ratio.
- **Layout**: Generous padding, potential for subtle rotation/box-shadows to mimic physical placement.
- `wake-style.css` file specifies the basic style.

### 4. Formatting

Use all formatting true to alan wake style.
