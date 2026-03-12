from openai import OpenAI
import base64
import os
import fitz  # PyMuPDF
import mimetypes

# Configured by environment variables
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

def encode_image(image_path):
    """Encodes a file from disk to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def transcribe_image(base64_image, mime_type="image/png", extracted_images=None):
    """Sends the image to the Qwen model for transcription."""
    
    image_section = ""
    if extracted_images:
        img_list = "\n".join([f"- {os.path.basename(img)}" for img in extracted_images])
        image_section = f"""
**Images:**
The following images have been extracted from this page. If they are figures, charts, or diagrams relevant to the content, insert them at the appropriate location using `![Description](extracted_images/<filename>)`.
Do NOT include logos, icons, or decorative elements.
Available images:
{img_list}
"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": f"""Transcribe the main article body from this image into Markdown.

**Include:**
- The main text content
- **Tables**: Convert all tables into standard Markdown tables.
{image_section}

**Exclude (Do NOT transcribe):**
- The header at the top
- The entire metadata (e.g., "Target Article", citation details, dates, "Keywords", and the "What is Open Peer Commentary?" box)
- The footer (e.g., "© Cambridge University Press 2019", logos)

Output only the transcribed text."""
                }
            ]
        }
    ]

    chat_response = client.chat.completions.create(
        model="Qwen/Qwen3.5-4B",
        messages=messages,
        max_tokens=81920,
        temperature=1.0,
        top_p=0.95,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
        }, 
    )
    return chat_response.choices[0].message.content


def smart_join_pages(pages):
    if not pages:
        return ""
    
    merged_text = pages[0].strip()
    
    for page in pages[1:]:
        page = page.strip()
        if not page: continue
        
        # Check if the previous page ended with a hyphen (split word)
        if merged_text.endswith("-"):
            # Remove hyphen and join without space
            merged_text = merged_text[:-1] + page
        # Check if the previous page ended with sentence-ending punctuation
        elif merged_text[-1] not in ".!?\":":
            # Likely a mid-sentence split; join with a space
            merged_text += " " + page
        else:
            # Standard paragraph break
            merged_text += "\n\n" + page
            
    return merged_text

def extract_images_from_page(doc, page, page_index, output_dir="extracted_images"):
    """Extracts images from a PDF page and saves them."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image_list = page.get_images(full=True)
    saved_images = []
    
    for img_index, img in enumerate(image_list):
        xref = img[0]
        try:
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            image_filename = f"image_page{page_index + 1}_{img_index + 1}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)
            
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            # Return the path that the LLM should use
            saved_images.append(image_path)
        except Exception as e:
            print(f"Failed to extract image {img_index} on page {page_index}: {e}")
            
    return saved_images

def process_input(input_path, max_pages=None):
    """Determines file type and processes accordingly."""
    # Guess the MIME type based on extension
    mime_type, _ = mimetypes.guess_type(input_path)
    
    transcribed_texts = []
    
    # Check if file exists
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return None

    if mime_type == 'application/pdf':
        print(f"Processing PDF: {input_path}")
        try:
            doc = fitz.open(input_path)
            total_pages = max_pages if max_pages else len(doc)
            for i, page in enumerate(doc[:total_pages]):
                print(f"  - Processing page {i+1}/{total_pages}...")
                
                # Extract images from the page first
                page_images = extract_images_from_page(doc, page, i)
                if page_images:
                    print(f"    Extracted {len(page_images)} images.")

                # Render page to image (zoom=3 for high resolution ~216 DPI)
                zoom = 3.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                
                # Encode bytes directly without saving to disk
                base64_img = base64.b64encode(img_bytes).decode('utf-8')
                
                # Pass extracted images to transcribe function
                text = transcribe_image(base64_img, "image/png", extracted_images=page_images)
                transcribed_texts.append(text)
            doc.close()
        except Exception as e:
            print(f"Error processing PDF: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    elif mime_type and mime_type.startswith('image'):
        print(f"Processing Image: {input_path}")
        try:
            base64_img = encode_image(input_path)
            text = transcribe_image(base64_img, mime_type)
            transcribed_texts.append(text)
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
        
    else:
        print(f"Unsupported or unrecognized file type: {mime_type} for {input_path}")
        return None

    # Combine all pages using smart stitching
    return smart_join_pages(transcribed_texts)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe PDF or Image to Markdown using Qwen.")
    parser.add_argument("input_path", help="Path to the input file (PDF or Image).")
    parser.add_argument("-o", "--output", default="output.md", help="Path to the output Markdown file (default: output.md).")
    parser.add_argument('-m',"--max-pages", type=int, default=None, help="Maximum number of pages to process (for PDFs).")

    args = parser.parse_args()

    final_markdown = process_input(args.input_path, max_pages=args.max_pages)

    if final_markdown:
        with open(args.output, "w") as f:
            f.write(final_markdown)
        print(f"\nSuccessfully transcribed {args.input_path}.")
        print(f"Saved to {args.output}")
        
        # Preview the first 500 characters
        print("-" * 20)
        print(final_markdown[:500])