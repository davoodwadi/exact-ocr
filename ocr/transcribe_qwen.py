import base64
import os
import pymupdf  
import mimetypes
import gc
import os

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
from multiprocessing import cpu_count
from openai import OpenAI

from pydantic import BaseModel, Field
from typing import List, Optional

class Metadata(BaseModel):
    title: str = Field(description="The title of the paper. It must be a single line.")
    author: str = Field(description="Comma separated name of author(s). It must be a single line.")
    date: Optional[int] = Field(None, description="The year the paper was published, if provided.")
    journal: Optional[str] = Field(None, description="The name of the journal or conference where the paper was published, if any. It must be a single line.")

try:
    # Pydantic v2
    json_schema = Metadata.model_json_schema()
except AttributeError:
    # Pydantic v1
    json_schema = Metadata.schema()

# Global states
llm = None
lcpp_process = None
openai_client = None
LCPP_PORT = 8080

MODEL = {
    'name': "gemma-4-26B-A4B-it",
    "sampling_params": {
        'temp': 1.0,
        'top-p': 0.95,
        'top-k': 64
    }
}
MODEL_NAME = MODEL['name']

# MODEL_NAME = 'gemma-4-31B-it'
# MODEL_NAME = 'Qwen3.6-35B-A3B'
# MODEL_NAME = 'Nemotron-3-Nano-30B-A3B'

# MODEL_NAME = 'Qwen3.5-35B-A3B'
# MODEL_NAME = 'Qwen3.5-2B'
# MODEL_NAME = 'Qwen3.5-0.8B'
MAX_WORKERS = '4'

max_model_length = 20000
max_tokens = 16000

def get_model_name():
    if isinstance(MODEL, dict):
        return MODEL.get('name', 'gemma-4-26B-A4B-it')
    return MODEL

def get_sampling_params():
    if isinstance(MODEL, dict):
        return MODEL.get('sampling_params', {})
    return {}

def prepare_completion_kwargs(extra_body_override=None, default_temp=None):
    kwargs = {
        "model": get_model_name(),
        "max_tokens": max_tokens,
    }
    
    extra_body = {}
    if extra_body_override:
        extra_body.update(extra_body_override)
        
    sampling_params = get_sampling_params()
    
    for key, val in sampling_params.items():
        key_lower = key.lower().replace('-', '_')
        if key_lower in ('temp', 'temperature'):
            kwargs['temperature'] = val
        elif key_lower in ('top_p', 'topp'):
            kwargs['top_p'] = val
        elif key_lower == 'presence_penalty':
            kwargs['presence_penalty'] = val
        elif key_lower == 'frequency_penalty':
            kwargs['frequency_penalty'] = val
        else:
            extra_body[key_lower] = val
            
    if 'temperature' not in kwargs and default_temp is not None:
        kwargs['temperature'] = default_temp
        
    if extra_body:
        kwargs['extra_body'] = extra_body
        
    return kwargs

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_free_port(start_port=8000):
    port = start_port
    while is_port_in_use(port):
        port += 1
    return port

def attach_existing_llama_cpp_server(port=8080):
    """Check if a llama-server is already running on the given port and attach to it."""
    global openai_client, LCPP_PORT, MODEL_NAME, MODEL
    if not is_port_in_use(port):
        return False
    try:
        client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")
        models = client.models.list()
        running_model = models.data[0].id if models.data else None
        if running_model:
            print(f"Found existing llama-server on port {port} running model '{running_model}'. Attaching to it.")
            LCPP_PORT = port
            MODEL_NAME = running_model
            if isinstance(MODEL, dict):
                MODEL['name'] = running_model
            else:
                MODEL = {'name': running_model, 'sampling_params': {}}
            openai_client = client
            return True
    except Exception:
        pass
    return False

def llama_cpp_logs():
    logfile = '/tmp/llama-server-8080.log'
    with open(logfile, encoding='utf-8', errors='replace') as f:
        logs = f.read()
    return logs
def start_llama_cpp_server():
    global lcpp_process, openai_client, LCPP_PORT, MAX_WORKERS

    if lcpp_process is not None:
        return

    kill_existing_llama_servers()
    # Reuse an already-running server if available on the default port
    # if attach_existing_llama_cpp_server(8080):
    #     return

    LCPP_PORT = find_free_port(8080)
    model_name = get_model_name()

    print(f"Starting llama-server on port {LCPP_PORT} for model {model_name}...")

    env = os.environ.copy()
    cache_dir = os.environ.get('LLAMA_CACHE')
    cache_dir_path = Path(cache_dir)
    model_dir = next(p for p in cache_dir_path.iterdir() if model_name in p.name)
    # print('model_dir', list(model_dir.iterdir()))
    # print()
    model_info=dict()
    for p in model_dir.iterdir():
        if 'mmproj' in p.name:
            model_info['mmproj'] = p.as_posix()
        elif model_name in p.name:
            model_info['MODEL_NAME'] = p.as_posix()
    print('model_info', model_info)

    cmd = [ 
        os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
        "-m", model_info['MODEL_NAME'],
        "--port", str(LCPP_PORT),
        '-ngl', '99',
        '--threads', str(cpu_count()),
        "--reasoning", 'off',
        '--flash-attn', 'on',
    ]
    if model_info.get('mmproj'):
        cmd.extend([
            "--mmproj", model_info['mmproj'],
        ])

    if 'gemma-4' in model_name.lower():
        cmd.extend([
            '--image-min-tokens', '1120', 
            '--image-max-tokens', '1120', 
            '--ubatch-size', str(4096), 
            '--cache-type-k', 'q4_0',
            '--cache-type-v', 'q4_0',
            '--batch-size', str(4096),

            # '--ctx-size', '14192',
            '--ctx-size', '15000',
        ])
        if '31b' in model_name.lower():
            MAX_WORKERS = '2'

    cmd.extend([
    "--parallel", MAX_WORKERS,
    ])


    print('cmd', cmd)
    # exit()

    # Redirect server logs to a file to avoid cluttering the terminal
    lcpp_log_path = Path(f"/tmp/llama-server-{LCPP_PORT}.log")
    lcpp_log_file = open(lcpp_log_path, "w")
    print(f"llama-server logs → {lcpp_log_path}")
    lcpp_process = subprocess.Popen(cmd, env=env, stdout=lcpp_log_file, stderr=lcpp_log_file)
    
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
                print(llama_cpp_logs())
                sys.exit(1)
            time.sleep(4)
            
    if not ready:
        print("llama-server failed to start within the timeout period.")
        stop_llama_cpp_server()
        sys.exit(1)
        
    print("llama-server is ready!")
    print(llama_cpp_logs())

def kill_existing_llama_servers():
    """Finds and kills any hanging llama-server processes from previous runs."""
    print("Ensuring no old llama-server processes are running...")
    try:
        # forcefully kill any process with 'llama-server' in its command line
        subprocess.run(["pkill", "-9", "-f", "llama-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Give the OS a moment to free up the ports
        time.sleep(1)
    except Exception as e:
        print(f"Warning: Could not kill existing servers: {e}")

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


def encode_image(image_path):
    """Encodes a file from disk to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def prepare_page_messages(base64_image, mime_type="image/png", extracted_images=None):
    """Prepares the message list for a single page."""
    
    image_section = ""
    extracted_images = [Path(ei).name for ei in extracted_images]

    if extracted_images:
        img_list = "\n".join([f"- {os.path.basename(img)}" for img in extracted_images])
        image_section = f"""
**Images:**
The following images have been extracted from this page. If they are figures, charts, or diagrams relevant to the content, insert them at the appropriate location using `![Description](<filename>)`.
Do NOT include logos, icons, or decorative elements.
Available images:
{img_list}
"""


    user_message = f"""Transcribe the main article body from this image.

**Include:**
- The main text content
- **Tables**: Convert all tables into standard Markdown tables.
{image_section}

**Exclude (Do NOT transcribe):**
- The page metadata (e.g., author biographies, footnotes, etc.)

Remember to only transcribe the main body of the paper, verbatim.
Output ONLY the final transcribed text, exactly as it appears. Do not include explanations or any other text."""

    messages = [
        # {
        #     'role':'system', 'content':system_message
        # },
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
                    "text": user_message
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

def get_full_text(doc):
    full_text_for_metadata = ''
    for i, page in enumerate(doc):
        full_text_for_metadata += page.get_text() + "\n"
        # Optimization: Stop if we have enough text
        if len(full_text_for_metadata) > 2000:
            break
    return full_text_for_metadata

def get_first_n_page_images(doc, n=2):
    images = []
    for i, page in enumerate(doc):
        if i<n:
            zoom = 3.0
            mat = pymupdf.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            # Encode bytes directly without saving to disk
            page_image = base64.b64encode(img_bytes).decode('utf-8')
            images.append(page_image)
    return images

def extract_metadata_openai_api(text, images):
    """
    Extracts metadata from the text using the OpenAI API.
    """
    num_pages = len(images)
    # print("Extracting metadata (Title/Author) via OpenAI API...")
    system_prompt = f'''Extract the metadata from the provided pdf. 
    The image of {num_pages} pages from the same paper have been provided. 
    For the journal field, only output the name of the journal or conference, which is only a few words.
    Each entry MUST be on a single line.
    Exactly follow the json schema for your output. 
    Do not output any other text.'''
    messages = [{"role": "system", "content": system_prompt}]
    
    user_content = []
    # 2. Append each image to the same user message content
    for base64_image in images:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
        })
    # user_content.append({
    #     "type": "text",
    #     "text": f"{text}"
    # }) 
    messages.append({'role':'user', 'content':user_content})    
    
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "paper-metadata",
                    "schema": json_schema,
                }
            },
            extra_body={"chat_template_kwargs": {"enable_thinking": False},},
            max_tokens=max_tokens,
        )
        generated_text = response.choices[0].message.content.strip()
        print(f"json output metadata:\n{generated_text}")
        metadata_dict = json.loads(generated_text)
        clean_data = {k: v[:200] for k, v in metadata_dict.items() if v}

        # print('generated_text metadata', generated_text)            
        return clean_data
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'response') and hasattr(e.response, 'json'):
            try:
                error_msg += f" - Response: {e.response.json()}"
            except Exception:
                error_msg += f" - Response (raw): {e.response.text}"
        print(f"Error extracting metadata via OpenAI API: {error_msg}")
        return {}

def transcribe_single_page_openai(args):
    page_idx, message = args
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=message,
            max_tokens=max_tokens,
            temperature=0.,
            # temperature=0.7,
            # top_p=0.8,
            # presence_penalty=1.5,
            # extra_body={
            #     "top_k": 20,
            #     # "chat_template_kwargs": {"enable_thinking": False},
            # },
        )
        # print(response.choices[0].message.content)
        if response.choices[0].finish_reason!='stop':
            print(f'Page {page_idx} incomplete with {response.choices[0].finish_reason}')
        print(f"Page {page_idx} finish reason: {response.choices[0].finish_reason}")
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'response') and hasattr(e.response, 'json'):
            try:
                error_msg += f" - Response: {e.response.json()}"
            except Exception:
                error_msg += f" - Response (raw): {e.response.text}"
        print(f"Error transcribing page {page_idx} via OpenAI API: {error_msg}")
        return ""

def transcribe_pages_openai_api(messages_batch, input_path):
    print(f"Running batch inference on {len(messages_batch)} pages...")
    transcribed_texts = []
    
    # Prepare arguments with page numbers
    items = [(i + 1, msg) for i, msg in enumerate(messages_batch)]
    
    # Use ThreadPoolExecutor to run requests concurrently
    with ThreadPoolExecutor(max_workers=int(MAX_WORKERS)) as executor:
        results = list(tqdm(executor.map(transcribe_single_page_openai, items), total=len(items), desc=f"Transcribing {input_path.name}"))
        
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
    parser.add_argument('-t', "--text-only", action="store_true", help="Bypass LLM and extract raw text from PDF directly")
    parser.add_argument('-s', "--size", type=str, default="large", help="Size of LLM used: 'large' | 'small'")
    args = parser.parse_args()

    if args.size=='small':
        MODEL_NAME = 'Qwen3.5-2B'

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
                    start_llama_cpp_server()
                    # stop_llama_cpp_server()
                    # print('stopped the server')
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
                    doc = pymupdf.open(input_path)
                    total_pages = args.max_pages if args.max_pages else len(doc)

                    if not args.text_only:
                        # Extract metadata using first 1000 chars of full text
                        print("Extracting text for metadata analysis...")
                        full_text_for_metadata = get_full_text(doc) 
                        first_n_images_list = get_first_n_page_images(doc, n = 2)
                        clean_data = extract_metadata_openai_api(text=full_text_for_metadata, images=first_n_images_list)
                            
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
                            mat = pymupdf.Matrix(zoom, zoom)
                            pix = page.get_pixmap(matrix=mat)
                            # --- NEW CODE: Save a sample of the first page ---
                            # if i == 0:
                                # sample_path = f"sample_page_{i+1}_resolution.png"
                                # pix.save(sample_path)
                                # print(f"Saved resolution sample to {sample_path}")
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
                            start_llama_cpp_server()
                        except Exception as e:
                            print(f"Failed to initialize server: {e}")
                            sys.exit(1)
                            
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
                kindle_style_file = Path(__file__).resolve().parent.parent / 'kindle-style.css'        
                EPUB_CMD = f'pandoc "{output_path_md}" -o "{output_path_epub}" --css "{kindle_style_file.as_posix()}" --resource-path=".:{images_dir}" --metadata lang="en-US" --toc'
                # EPUB_CMD = f"pandoc {output_path} -o {epub_path} --css {wake_style_file.as_posix()} --resource-path=.:{images_dir}"
                print(f"Executing: {EPUB_CMD}")
                try:
                    subprocess.run(EPUB_CMD, shell=True, check=True)
                    print(f"Successfully generated EPUB: {output_path_epub}")
                except subprocess.CalledProcessError as e:
                    print(f"Error generating EPUB with pandoc: {e}")
        else:
            print(f"Warning: {wake_style_file} not found. Skipping PDF generation.")


