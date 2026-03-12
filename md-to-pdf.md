python extract_pdf.py rational.pdf

pandoc input.md -o manuscript.pdf --css wake-style.css --pdf-engine=weasyprint --resource-path=.:images:extracted_images -V lang=en-US