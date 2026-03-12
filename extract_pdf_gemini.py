import os
import sys
import argparse
import fitz  # PyMuPDF
from google import genai
from google.genai import types

def extract_images(pdf_path, output_dir):
    """
    Extracts images from the PDF and saves them to the output directory.
    Returns a dictionary mapping page numbers (0-indexed) to a list of image filenames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    doc = fitz.open(pdf_path)
    image_map = {}
    
    print(f"Extracting images from {pdf_path}...")
    
    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        
        if image_list:
            image_map[page_index] = []
            print(f"  Found {len(image_list)} images on page {page_index + 1}")
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                image_filename = f"image_page{page_index + 1}_{img_index + 1}.{image_ext}"
                image_path = os.path.join(output_dir, image_filename)
                
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                image_map[page_index].append(image_filename)
                
    return image_map

def extract_content():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract text and tables from PDF to Markdown using Gemini.')
    parser.add_argument('input_pdf', help='Path to the input PDF file')
    parser.add_argument('-o', '--output', default='input.md', help='Path to the output Markdown file (default: input.md)')
    parser.add_argument('--model', default='gemini-3-flash-preview', help='Gemini model to use (default: gemini-2.0-flash)')
    parser.add_argument('--image-dir', default='images', help='Directory to save extracted images (default: images)')
    
    args = parser.parse_args()
    
    # Initialize client using environment variable for API key
    # Ensure GEMINI_API_KEY or GOOGLE_API_KEY is set in your environment
    try:
        client = genai.Client()
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        print("Please ensure GOOGLE_API_KEY or GEMINI_API_KEY environment variable is set.")
        sys.exit(1)
    
    if not os.path.exists(args.input_pdf):
        print(f"Error: {args.input_pdf} not found.")
        sys.exit(1)

    # Step 1: Extract images
    try:
        image_map = extract_images(args.input_pdf, args.image_dir)
    except Exception as e:
        print(f"Error extracting images: {e}")
        # Continue without images if extraction fails
        image_map = {}

    # Prepare image context for the prompt
    image_context = "I have extracted the following images from the PDF. Please insert them into the Markdown at the appropriate locations using the syntax `![Description](images/filename)`:\n"
    if image_map:
        for page_idx, filenames in image_map.items():
            image_context += f"- Page {page_idx + 1}: {', '.join(filenames)}\n"
    else:
        image_context += "No images were extracted programmatically. If you see images in the PDF visually, use a placeholder like `*[Image: Description]*`.\n"

        
    print(f"Reading {args.input_pdf}...")
    try:
        with open(args.input_pdf, "rb") as f:
            pdf_bytes = f.read()
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        sys.exit(1)

    print(f"Sending request to Gemini ({args.model})...")
    try:
        prompt = f"""
        Extract the main content from this PDF into clean Markdown format. Focus on the article body.
        
        {image_context}
        
        Requirements:
        1. TEXT: Preserve the main article text. Format headings (#, ##) and bold/italic text to reflect the document structure.
        2. TABLES: Convert all tables into standard Markdown tables.
        3. IMAGES: Use the provided image filenames where appropriate. 
           - CRITICAL: Do NOT include images that are logos, icons, buttons (e.g., "Check for updates"), or decorative elements.
           - Only include images that are figures, charts, or diagrams relevant to the article content.
        4. EXCLUSIONS: Do NOT include:
           - Metadata (dates, "Target Article Accepted", "Keywords", etc.)
           - Copyright notices (e.g. "© Cambridge University Press")
           - Publisher logos or branding images
           - Author affiliations (superscripts like <sup>a</sup>)
           - Running headers/footers (e.g. page numbers, "Lieder and Griffiths: ...")
           - "What is Open Peer Commentary?" sections.
           - "Check for updates" logos or buttons.
        5. Do not include any introductory or concluding remarks. Just output the clean content.
        """
        
        response = client.models.generate_content(
            model=args.model,
            contents=[
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type='application/pdf',
                ),
                prompt
            ]
        )
        
        if response.text:
            print(f"Writing extracted content to {args.output}...")
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(response.text)
            print("Done.")
        else:
            print("Error: No text returned from the model.")
            
    except Exception as e:
        print(f"An error occurred during generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    extract_content()
