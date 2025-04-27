Monument Image Generation from Textual Prompts This project demonstrates a complete, production-level pipeline for generating realistic and detailed images of monuments based on textual descriptions (text-to-image generation). It integrates data preprocessing, retrieval-augmented generation (RAG) techniques, embedding with FAISS, LoRA fine-tuning of a language model, and Stable Diffusion for image generation.
Project Structure The workflow is structured into the following main stages:


**1.	Data Preprocessing: Matching Images and Descriptions Directories:**
image_dir: Contains .jpg images of monuments.
desc_dir: Contains .txt descriptions for each image.
Matching Process:
Extract filenames (prefixes) and match each image with its corresponding description.
Build a paired dataset of {image_path, description}.


**2.	Text Chunking for RAG (Retrieval-Augmented Generation) Tool: langchain's RecursiveCharacterTextSplitter**
Objective:
Split long descriptions into smaller text chunks (512 characters with 50 overlap).
Facilitates efficient retrieval during the prompt understanding.



**3.	Text Embedding and Vector Store Construction (FAISS) Tool: Sentence-Transformers (all-MiniLM-L6-v2) for encoding text.**
Vector Store:
Use FAISS (IndexFlatL2) to store and index the description embeddings.
Retrieval:
Given a new textual query, retrieve the most semantically similar descriptions using FAISS search.



**4.	LoRA Fine-Tuning of a Language Model (PEFT) Base Model: mistralai/Mistral-7B-v0.1**
Fine-Tuning Technique: PEFT (Parameter-Efficient Fine-Tuning) with LoRA (Low-Rank Adaptation).
Configuration:
r=8, lora_alpha=32, lora_dropout=0.1, bias=none, task_type=CAUSAL_LM
Goal:
Adapt the base LLM with fewer parameters for better understanding and generation related to monument reconstructions.


**5.	Image Generation Using Stable Diffusion Model: runwayml/stable-diffusion-v1-5**
Pipeline: diffusers StableDiffusionPipeline
Hardware:
Runs on CUDA-enabled GPUs for efficient generation.
Prompts:
Users can input custom prompts describing the monument.
Example prompts include detailed architectural features (e.g., arches, sandstone, pagoda roofs, Roman temples).
Outputs:
High-quality, realistic .jpg images saved locally.

**Installation**
pip install pillow langchain sentence-transformers faiss-cpu peft transformers diffusers huggingface_hub 

**Additional setup for Hugging Face models:**
from huggingface_hub import login
login() # Enter your Hugging Face token

**Key Technologies Used**
-------------------------------------------------------------------------------------------------------------------------------------
Technology                     |Purpose

PIL (Pillow)                  | Image loading and saving 

Langchain                     | Text chunking for RAG

Sentence-Transformers         | Text embedding 

FAISS                         |Vector similarity search 

PEFT + LoRA                   |Efficient LLM fine-tuning 

Stable Diffusion (Diffusers)  | Text-to-Image generation

Hugging Face Hub              |Model management 



**Example Usage**
After setting up everything, generate a new monument image:
prompt = "A majestic ancient monument with intricate carvings and large stone columns, partially reconstructed blending historical architecture with modern elements." 
image = pipe(prompt).images[0]
image.save("reconstructed_monument.jpg") 



**Example generated images include:**
A reddish-pink observatory with sundials.
A majestic stone castle overlooking the sea.
A Roman temple with grand arches.
A Buddhist temple nestled in the mountains.



**Full Pipeline Diagram:** 


[Image + Description Data]

         ↓   
         
[Text Chunking (RAG)] 

         ↓ 
         
[Text Embedding (MiniLM)]

         ↓ 
[Vector Index (FAISS)] 

         ↓ 
         
[Prompt Retrieval + Augmentation] 

         ↓ 
         
[Fine-tuned LLM (Mistral 7B + LoRA)] 

         ↓
         
[Stable Diffusion Image Generation] 

         ↓          
         
[Generated Monument Image] 



**Important Notes Data Sources:**
Dataset should have a one-to-one mapping between images and textual descriptions.
GPU Requirement: Image generation via Stable Diffusion requires a CUDA-capable GPU for fast generation.
Model Access:
Ensure you have access to the base models (Mistral, Stable Diffusion) via Hugging Face authentication.
Scalability:
The pipeline can be expanded to larger datasets and more fine-grained RAG retrieval strategies for even better prompt understanding.

