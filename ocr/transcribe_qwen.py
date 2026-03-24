from vllm import LLM, SamplingParams
import base64
import os
import fitz  # PyMuPDF
import mimetypes
import gc
import torch
import os

# To help avoid fragmentation and out of memory issues
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
import argparse
import subprocess
import json
import textwrap
import json
import yaml
from tqdm import tqdm
import time
import socket
import atexit
import sys
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

from vllm.sampling_params import StructuredOutputsParams
from pydantic import BaseModel
from typing import List, Optional
from pydantic import Field

class Metadata(BaseModel):
    title: str = Field(description="Title of the paper")
    author: str = Field(description="Name of author")
    date: Optional[str] = Field(None, description="Publication date in YYYY-MM-DD format if available, otherwise YYYY")
    keywords: Optional[List[str]] = Field(None, description="List of keywords")
    journal: Optional[str] = Field(None, description="Journal or Conference name")

try:
    # Pydantic v2
    json_schema = Metadata.model_json_schema()
except AttributeError:
    # Pydantic v1
    json_schema = Metadata.schema()

structured_outputs_params = StructuredOutputsParams(json=json_schema)

# Global states
llm = None
lcpp_process = None
openai_client = None
LCPP_PORT = 8080

MODEL_NAME = 'Qwen3.5-35B-A3B'
# MODEL_NAME = 'Qwen3.5-0.8B'

max_model_length = 20000
max_tokens = 16000

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_free_port(start_port=8000):
    port = start_port
    while is_port_in_use(port):
        port += 1
    return port

def start_llama_cpp_server():
    global lcpp_process, openai_client, LCPP_PORT
    
    if lcpp_process is not None:
        return
        
    LCPP_PORT = find_free_port(8080)
    
    print(f"Starting llama-server on port {LCPP_PORT} for model {MODEL_NAME}...")

    env = os.environ.copy()
    cache_dir = os.environ.get('LLAMA_CACHE')
    cmd = [
        os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
        "-m", f"{cache_dir}/{MODEL_NAME}-GGUF/{MODEL_NAME}-UD-Q4_K_XL.gguf",
        "--mmproj", f"{cache_dir}/{MODEL_NAME}-GGUF/mmproj-BF16.gguf",
        "--port", str(LCPP_PORT)
    ]
    
    # Start process in background, allowing stderr to print to console so user can see server crashes
    lcpp_process = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=sys.stderr)
    
    # Wait for server to be ready
    openai_client = OpenAI(base_url=f"http://localhost:{LCPP_PORT}/v1", api_key="EMPTY")
    
    print("Waiting for llama-server to be ready...")
    max_retries = 60 # 2 minutes
    ready = False
    for i in range(max_retries):
        try:
            # Check models endpoint to verify it's up
            openai_client.models.list()
            ready = True
            break
        except Exception:
            if lcpp_process.poll() is not None:
                print("llama-server process terminated unexpectedly.")
                sys.exit(1)
            time.sleep(4)
            
    if not ready:
        print("llama-server failed to start within the timeout period.")
        stop_llama_cpp_server()
        sys.exit(1)
        
    print("llama-server is ready!")

def stop_llama_cpp_server():
    global lcpp_process
    if lcpp_process is not None:
        print("Stopping llama-server...")
        lcpp_process.terminate()
        try:
            lcpp_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            lcpp_process.kill()
        lcpp_process = None

# Register cleanup on exit
atexit.register(stop_llama_cpp_server)

def init_llm():
    global llm, MODEL_NAME
    if llm is None:
        print("Initializing VLLM...")
        llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=1,
            max_model_len=max_model_length, # Reduced from 32000
            gpu_memory_utilization=0.85,
            enforce_eager=True, # Save memory by not using CUDA graph
            trust_remote_code=True,
            # reasoning_parser="qwen3"
        )

def encode_image(image_path):
    """Encodes a file from disk to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def prepare_page_messages(base64_image, mime_type="image/png", extracted_images=None):
    """Prepares the message list for a single page."""
    
    image_section = ""
    extracted_images = [Path(ei).name for ei in extracted_images]
    # print(extracted_images)

    if extracted_images:
        img_list = "\n".join([f"- {os.path.basename(img)}" for img in extracted_images])
        image_section = f"""
**Images:**
The following images have been extracted from this page. If they are figures, charts, or diagrams relevant to the content, insert them at the appropriate location using `![Description](<filename>)`.
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
                    "text": f"""Transcribe the main article body from this image.

**Include:**
- The main text content
- **Tables**: Convert all tables into standard Markdown tables.
{image_section}

**Exclude (Do NOT transcribe):**
- The header at the top
- The entire metadata (e.g., "Target Article", citation details, dates, "Keywords", and the "What is Open Peer Commentary?" box)
- The footer (e.g., "© Cambridge University Press 2019", logos)

Output ONLY the final transcribed text. Do not include explanations or any other text."""
                }
            ]
        }
    ]
    return messages

def smart_join_pages(pages):
    if not pages:
        return ""
    
    merged_text = pages[0].strip()
    
    for i, page in enumerate(pages[1:], 2):
        page = page.strip()
        if not page: continue
        
        # merged_text+=f'\n\n# **Page: {i}**\n\n'

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

def cleanup_llm():
    """Explicitly clean up VLLM resources."""
    global llm
    if llm is not None:
        try:
            del llm
            gc.collect()
            torch.cuda.empty_cache()
            llm = None
        except Exception as e:
            print(f"Warning during cleanup: {e}")

def get_text_image(doc):
    full_text_for_metadata = ''
    first_page_image = None
    for i, page in enumerate(doc):
        if i==0:
            zoom = 3.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            # Encode bytes directly without saving to disk
            first_page_image = base64.b64encode(img_bytes).decode('utf-8')
            
        full_text_for_metadata += page.get_text() + "\n"
        # Optimization: Stop if we have enough text
        if len(full_text_for_metadata) > 2000:
            break
    return full_text_for_metadata, first_page_image

def extract_metadata_python_api(text, image):
    """
    Extracts metadata from the text using the Python API.
    """
    print("Extracting metadata (Title/Author) via Python API...")
    sampling_params_json = SamplingParams(
        temperature=0.7,
        top_p=1.0,
        top_k=40, 
        min_p=0.0,
        presence_penalty=2.0,
        repetition_penalty=1.0,
        max_tokens=max_tokens,
        structured_outputs=structured_outputs_params
    )
    system_prompt = 'Extract the metadata from the user text. Exactly follow the json schema for your output.'
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": text
                }
            ]
        },
    ]
    try:
        outputs = llm.chat(
            messages=messages, 
            sampling_params=sampling_params_json,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=False
        )
        generated_text = outputs[0].outputs[0].text.strip()
        print('num_tokens:',len(outputs[0].outputs[0].token_ids))
        if generated_text.startswith("```"):
            lines = generated_text.split('\n')
            if lines[0].startswith("```"): lines = lines[1:]
            if lines and lines[-1].startswith("```"): lines = lines[:-1]
            generated_text = "\n".join(lines).strip()
            
        return json.loads(generated_text)
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return {}

def extract_metadata_openai_api(text, image):
    """
    Extracts metadata from the text using the OpenAI API.
    """
    print("Extracting metadata (Title/Author) via OpenAI API...")
    system_prompt = 'Extract the metadata from the user text. Exactly follow the json schema for your output.'
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": text
                }
            ]
        },
    ]
    
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "Metadata",
                    "schema": json_schema,
                }
            },
            temperature=0.7,
            top_p=1.0,
            presence_penalty=2.0,
            extra_body={
                "top_k": 40,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            max_tokens=max_tokens,
        )
        generated_text = response.choices[0].message.content.strip()
        # print('generated_text metadata', generated_text)            
        return json.loads(generated_text)
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'response') and hasattr(e.response, 'json'):
            try:
                error_msg += f" - Response: {e.response.json()}"
            except Exception:
                error_msg += f" - Response (raw): {e.response.text}"
        print(f"Error extracting metadata via OpenAI API: {error_msg}")
        return {}

def transcribe_pages_python_api(messages_batch, input_path):
    print(f"Running VLLM batch inference on {len(messages_batch)} pages (Python API)...")
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        top_k=20,
        max_tokens=max_tokens
    )
    outputs = []
    for message in tqdm(messages_batch, desc=f"Transcribing {input_path.name}"):
        out = llm.chat(
            messages=message, 
            sampling_params=sampling_params, 
            use_tqdm=False
        )
        outputs.append(out[0])
        print('num_tokens:',len(out[0].outputs[0].token_ids))

    transcribed_texts = []
    for output in outputs:
        generated_text = output.outputs[0].text
        if "</think>" in generated_text:
            generated_text = generated_text.split("</think>")[-1].strip()
        transcribed_texts.append(generated_text)
    return transcribed_texts

def transcribe_single_page_openai(message):
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=message,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                # "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'response') and hasattr(e.response, 'json'):
            try:
                error_msg += f" - Response: {e.response.json()}"
            except Exception:
                error_msg += f" - Response (raw): {e.response.text}"
        print(f"Error transcribing page via OpenAI API: {error_msg}")
        return ""

def transcribe_pages_openai_api(messages_batch, input_path):
    print(f"Running VLLM batch inference on {len(messages_batch)} pages (OpenAI API)...")
    transcribed_texts = []
    
    # Use ThreadPoolExecutor to run requests concurrently
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(transcribe_single_page_openai, messages_batch), total=len(messages_batch), desc=f"Transcribing {input_path.name}"))
        
    for generated_text in results:
        if generated_text and "</think>" in generated_text:
            generated_text = generated_text.split("</think>")[-1].strip()
        transcribed_texts.append(generated_text or "")
        
    return transcribed_texts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe PDF or Image to Markdown using Qwen (VLLM).")
    parser.add_argument("input_paths", nargs='+', help="Paths to the input files (PDF or Image).")
    parser.add_argument("-e", "--epub", type=bool, default=False, help="Should produce epub?")
    parser.add_argument("-w", "--wake", type=bool, default=False, help="Should produce wake-style manuscript PDF file?")
    parser.add_argument('-m',"--max-pages", type=int, default=None, help="Maximum number of pages to process (for PDFs).")
    parser.add_argument('-f',"--force", action="store_true", help="Force retranscribe if md file exists")
    parser.add_argument('-P', "--python-api", action="store_true", help="Use VLLM Python API instead of spinning up an OpenAI server")
    parser.add_argument('-t', "--text-only", action="store_true", help="Bypass LLM and extract raw text from PDF directly")
    args = parser.parse_args()

    # Path to wake-style.css relative to this script
    wake_style_file = Path(__file__).resolve().parent.parent / 'wake-style.css'

    for input_file in args.input_paths:
        input_path = Path(input_file)
        output_stem = input_path.stem
        output_path_md = input_path.parent/f"{output_stem}.md"
        print(f'\nProcessing: {input_path}')
        print('output_path_md', output_path_md)

        if (not output_path_md.exists()) or args.force:
            if args.text_only:
                print('Extracting raw text (LLM bypassed)')
            else:
                print('Transcribing')
                # Init LLM early based on mode
                try:
                    if args.python_api:
                        init_llm()
                    else:
                        start_llama_cpp_server()
                except Exception as e:
                    print(f"Failed to initialize server: {e}")
                    sys.exit(1)

            # Guess the MIME type based on extension
            mime_type, _ = mimetypes.guess_type(input_path)

            messages_batch = []
            metadata = {}

            frontmatter = ""
            transcribed_texts = []
            
            if mime_type == 'application/pdf':
                print(f"Processing PDF: {input_path}")
                try:
                    doc = fitz.open(input_path)
                    total_pages = args.max_pages if args.max_pages else len(doc)

                    if not args.text_only:
                        # Extract metadata using first 1000 chars of full text
                        print("Extracting text for metadata analysis...")
                        full_text_for_metadata, first_page_image = get_text_image(doc) 
                        if args.python_api:
                            metadata_dict = extract_metadata_python_api(text=full_text_for_metadata, image=first_page_image)
                        else:
                            metadata_dict = extract_metadata_openai_api(text=full_text_for_metadata, image=first_page_image)
                            
                        clean_data = {k: v for k, v in metadata_dict.items() if v}
                        yaml_str = yaml.dump(clean_data, allow_unicode=True, default_flow_style=False, sort_keys=False)
                        frontmatter = f"---\n{yaml_str}---\n"
                        print(f"Metadata extracted:\n{frontmatter}")

                    print(f"Preparing {total_pages} pages...")
                    for i, page in enumerate(doc[:total_pages]):
                        # Extract images from the page first
                        page_images = extract_images_from_page(doc, page, i, output_dir=f"extracted_images_{output_stem}")
                        
                        if args.text_only:
                            # Direct text extraction via PyMuPDF
                            page_text = page.get_text()
                            
                            # Append image links manually at the bottom of the page text
                            if page_images:
                                for img_path in page_images:
                                    img_name = Path(img_path).name
                                    page_text += f"\n\n![Image]({img_name})\n\n"
                                    
                            transcribed_texts.append(page_text)
                        else:
                            # Render page to image (zoom=3 for high resolution ~216 DPI)
                            zoom = 2.0
                            mat = fitz.Matrix(zoom, zoom)
                            pix = page.get_pixmap(matrix=mat)
                            # --- NEW CODE: Save a sample of the first page ---
                            if i == 0:
                                sample_path = f"sample_page_{i+1}_resolution.png"
                                pix.save(sample_path)
                                print(f"Saved resolution sample to {sample_path}")
                            # ---------------------------------
                            img_bytes = pix.tobytes("png")
                            
                            # Encode bytes directly without saving to disk
                            base64_img = base64.b64encode(img_bytes).decode('utf-8')
                            # print('Size of image:', len(base64_img))
                            
                            # Prepare message for this page
                            messages = prepare_page_messages(base64_img, "image/png", extracted_images=page_images)
                            messages_batch.append(messages)
                            
                    doc.close()
                except Exception as e:
                    print(f"Error processing PDF: {e}")
                    import traceback
                    traceback.print_exc()

            elif mime_type and mime_type.startswith('image'):
                print(f"Processing Image: {input_path}")
                if args.text_only:
                    print("Warning: '-t / --text-only' flag is ignored for image files. Proceeding with LLM transcription.")
                    # Automatically disable text_only for images if it was set
                    args_text_only_was_true = True
                
                try:
                    base64_img = encode_image(input_path)
                    first_page_base64 = base64_img # It is the only page
                    messages = prepare_page_messages(base64_img, mime_type)
                    messages_batch.append(messages)
                except Exception as e:
                    print(f"Error processing image: {e}")
                
            else:
                print(f"Unsupported or unrecognized file type: {mime_type} for {input_path}")

            # Run Inference if we aren't in text_only mode for PDFs, or if it's an image
            if not args.text_only or (mime_type and mime_type.startswith('image')):
                if messages_batch:
                    # If we bypassed LLM earlier but hit an image, we need to initialize it now
                    if args.text_only and llm is None and openai_client is None:
                        try:
                            if args.python_api:
                                init_llm()
                            else:
                                start_llama_cpp_server()
                        except Exception as e:
                            print(f"Failed to initialize server: {e}")
                            sys.exit(1)
                            
                    if args.python_api:
                        llm_transcribed_texts = transcribe_pages_python_api(messages_batch, input_path)
                    else:
                        llm_transcribed_texts = transcribe_pages_openai_api(messages_batch, input_path)
                        
                    if not args.text_only:
                        transcribed_texts = llm_transcribed_texts
                    else:
                        # Append image transcriptions to texts if any
                        transcribed_texts.extend(llm_transcribed_texts)

            # Combine all pages using smart stitching
            final_markdown = smart_join_pages(transcribed_texts)

            if (mime_type == 'application/pdf') and final_markdown and frontmatter:
                final_markdown = frontmatter + "\n\n" + final_markdown

            if final_markdown:
                with open(output_path_md, "w") as f:
                    f.write(final_markdown)
                print(f"\nSuccessfully transcribed {input_path}.")
                print(f"Saved to {output_path_md}")
        else:
            print(f'already exists: {output_path_md}')


        if (wake_style_file).exists():
            output_path_pdf = input_path.parent/f"{output_stem}_manuscript.pdf"

            images_dir = os.path.abspath(f"extracted_images_{output_stem}") 
            print('images_dir', images_dir)
            if args.wake:
                CMD = f"pandoc {output_path_md} -o {output_path_pdf} --css {wake_style_file.as_posix()} --pdf-engine=weasyprint --pdf-engine-opt=--base-url={images_dir}/ -V lang=en-US"
                print(f"Executing: {CMD}")
                try:
                    subprocess.run(CMD, shell=True, check=True)
                    print(f"Successfully generated PDF: {output_path_pdf}")
                except subprocess.CalledProcessError as e:
                    print(f"Error generating PDF with pandoc: {e}")
            if args.epub:
                output_path_epub = input_path.parent/f"{output_stem}.epub"        
                EPUB_CMD = f'pandoc "{output_path_md}" -o "{output_path_epub}" --css "{wake_style_file.as_posix()}" --resource-path=".:{images_dir}" --metadata lang="en-US"'
                # EPUB_CMD = f"pandoc {output_path} -o {epub_path} --css {wake_style_file.as_posix()} --resource-path=.:{images_dir}"
                print(f"Executing: {EPUB_CMD}")
                try:
                    subprocess.run(EPUB_CMD, shell=True, check=True)
                    print(f"Successfully generated EPUB: {output_path_epub}")
                except subprocess.CalledProcessError as e:
                    print(f"Error generating EPUB with pandoc: {e}")
        else:
            print(f"Warning: {wake_style_file} not found. Skipping PDF generation.")



    # Ensure cleanup on exit for python API
    if args.python_api:
        cleanup_llm()
