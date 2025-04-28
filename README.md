# README

# Overview

This project leverages the power of text-to-image generation to create new images of monuments based on user prompts. By combining advanced deep learning techniques such as Stable Diffusion and semantic embedding models, this system can generate high-quality artistic images that are influenced by a specific artist's style. Whether you're interested in visualizing a monumental structure or exploring the fusion of different art styles, this tool can serve as an inspiring creative resource.

# Requirements

Before diving into the implementation, you will need to install the necessary libraries. The following installations should be performed to set up the environment:

!pip install diffusers transformers accelerate torch torchvision torchaudio sentence-transformers faiss-cpu bitsandbytes peft --quiet

!pip install huggingface_hub

These packages provide everything needed to run the model, including libraries for text embedding, image transformation, and the diffusion model.

# Components of the Code

# Imports

The code imports several essential libraries and modules for different tasks:

Torch for deep learning models.

diffusers and transformers for accessing pre-trained models and pipelines.

Sentence-Transformers for creating embeddings from text descriptions.

Faiss for efficient retrieval of similar descriptions from a large dataset.

PEFT for advanced fine-tuning with LoRA (Low-Rank Adaptation).

Huggingface_hub to interact with the Hugging Face model repository.

# Configuration

The configuration section sets up various paths and devices used in the pipeline:

dataset_root: Path to the folder containing images of artworks.

description_root: Path to the folder with corresponding textual descriptions.

device: Automatically selects either GPU (cuda) or CPU based on availability.

embedding_model_name: Specifies the pre-trained model used for embedding textual descriptions.

base_model_name: The name of the pre-trained dreamlike-diffusion model optimized for generating artistic images.

# Embedding Model Setup

A pre-trained Sentence Transformer (all-MiniLM-L6-v2) is loaded to convert the text descriptions into fixed-length embeddings. These embeddings capture the semantic content of the text, which will be used to retrieve similar descriptions from a dataset

embedding_model = SentenceTransformer(embedding_model_name).to(device)

# Diffusion Pipeline

The heart of the image generation lies in the Stable Diffusion Pipeline. It uses a pre-trained model (dreamlike-diffusion-1.0) to convert the text prompts into images. This model is fine-tuned for generating artistic content, making it an excellent choice for style-based image generation.

pipe = StableDiffusionPipeline.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    use_safetensors=True,
).to(device)

The pipeline is then used to generate images by passing user prompts.

# Data Preparation

To prepare the dataset for training and retrieval, images and their corresponding descriptions are loaded and processed:

 * Descriptions are stored in text files (.txt).

* Images are stored in .jpg format.

* The prepare_dataset function combines the images and descriptions into a structured dataset, transforming the images into tensors and encoding the descriptions.

def get_image_description_pairs():
    # Load description files and match them with images.

# FAISS Index

Once the dataset is prepared, FAISS (Facebook AI Similarity Search) is used to create an efficient index of textual descriptions. This allows for fast retrieval of similar descriptions given a user query.

desc_embeds = embedding_model.encode(dataset["chunk"], convert_to_numpy=True)
index = faiss.IndexFlatL2(desc_embeds.shape[1])
index.add(desc_embeds)

# Similarity Retrieval

Given a user query (e.g., an artist's name), the system retrieves the top k most similar descriptions from the dataset. This helps in tailoring the prompt to the user’s specifications and artist influence.

def retrieve_similar_chunks(query, top_k=5):
    query_emb = embedding_model.encode([query])
    D, I = index.search(query_emb, top_k)
    return [dataset[int(i)]["chunk"] for i in I[0]]

# Smart Prompt Fusion

To ensure that the final prompt is creatively diverse and semantically aligned with the user's input, Smart Prompt Fusion combines the user’s description with randomly selected style-related phrases from the most similar descriptions retrieved earlier. The result is a final prompt that blends the user input with stylistic elements of the selected artist.

def create_final_prompt(user_prompt, artist_name, retrieved_chunks):
    style_descriptions = ". ".join(random.sample(retrieved_chunks, min(4, len(retrieved_chunks))))
    final_prompt = f"{user_prompt}, captured in the essence of {artist_name}'s style. Emphasized artistic traits include: {style_descriptions}."


# Image Generation

The generate_image function uses the constructed prompt to generate an image of the described monument in the specified artist's style. The Stable Diffusion model processes the prompt and returns an image.

def generate_image(user_prompt, artist_name, show=True):
    retrieved_chunks = retrieve_similar_chunks(artist_name)
    final_prompt = create_final_prompt(user_prompt, artist_name, retrieved_chunks)
    image = pipe(final_prompt).images[0]

# LoRA Fine-Tuning (Optional)

For further customization, LoRA (Low-Rank Adaptation) can be applied to fine-tune the U-Net model, improving its ability to generate images in specific styles. This is an advanced feature for users looking to modify the pre-trained model.

def prepare_lora(pipe):
    lora_config = LoraConfig(r=8, lora_alpha=16)
    unet = pipe.unet
    unet = prepare_model_for_kbit_training(unet)
    lora_unet = get_peft_model(unet, lora_config)

# Testing and Saving Generated Images

Several test prompts are provided to demonstrate the image generation process. The generated images are saved and displayed for the user to review:

user_prompt = "A majestic tree with intricate details"
artist_name = "Leonardo da Vinci"
image, prompt_used = generate_image(user_prompt, artist_name)
image.save("/kaggle/working/tree_leonardo_style.png")

# How It Works

User Input: A user provides a text prompt describing the scene or monument they want to generate.

Artist Style Selection: The user selects an artist’s name to influence the visual style of the generated image.

Textual Embedding: The system encodes the description into a semantic embedding.

Similarity Search: The system retrieves the most similar descriptions from the dataset using FAISS.

Smart Prompt Fusion: The retrieved descriptions are fused with the user’s prompt to create a final detailed prompt.

Image Generation: The final prompt is passed through the Stable Diffusion model to generate an image.

Customization (Optional): Users can fine-tune the model using LoRA for more specific styles or features.


# Conclusion

This system enables creative image generation based on textual prompts while allowing users to imbue their images with the styles of famous artists. It's ideal for projects involving the reconstruction of monuments, artistic reinterpretation, or any task requiring the fusion of descriptive text and artistic imagery. Whether you're working on a creative project or simply exploring the intersection of art and AI, this tool offers powerful capabilities for text-to-image generation.
















