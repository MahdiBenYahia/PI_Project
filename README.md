# ArtRevive

## Overview
ArtRevive is a web‑based Heritage Restoration Platform designed to preserve, restore, and reimagine cultural artifacts and sites through cutting‑edge AI. Developed as part of the **Esprit School of Engineering** curriculum, ArtRevive brings together deep learning, transformer architectures, convolutional neural networks, generative AI (RAG), diffusion models, and 3D graphics in a seamless, interactive experience. Whether you need to detect forgeries, inpaint missing sculpture fragments, auto‑generate virtual reconstructions, or convert 2D restorations into fully navigable 3D models, ArtRevive has you covered.

## Features
- **Fake vs. Real Painting Detection**  
  - Leverages CNNs and Vision Transformers (ViT) to classify and flag suspect or forged artworks.  
- **Statue & Monument Restoration**  
  - Applies diffusion‑based image inpainting to fill cracks, holes, and missing fragments in photographs of sculptures and edifices.  
- **Site Generation**  
  - Uses diffusion models and prompts to generate heritage sites .  
- **Style‑Transfer Painting Generator**  
  - produce new artworks in the style of masters like Monet, Van Gogh, and Da Vinci.  
- **Art Assistant (Q&A)**  
  - A Retrieval‑Augmented Generation (RAG) pipeline built on Hugging Face Transformers to answer questions about art history, techniques, provenance, and more.  
- **2D → 3D Conversion**  
  - Combines depth estimation networks and mesh‑reconstruction algorithms to turn 2D images of restored statues into fully textured, interactive 3D models.

### Frontend
- **React.js** + **Three.js** for dynamic 3D visuals and scene management  
- **TypeScript** for type safety and developer productivity  
- **Tailwind CSS** for utility‑first, responsive styling  

### Backend
- **Django** as the core web framework, handling routing, authentication, and the admin interface  
- **Django REST Framework** for building robust, versioned API endpoints  

### AI & Deep Learning
- **TensorFlow** & **PyTorch** 
- **Convolutional Neural Networks (CNNs)**  
- **Vision Transformers (ViT)**  
- **Generative Diffusion Models** 
- **Retrieval‑Augmented Generation (RAG)**   
- **Depth‑Estimation Networks**   


### Other Tools
- **OpenCV** for image preprocessing (resizing, normalization, contour detection)  
- **Blender** for manual mesh cleanup and asset refinement  
- **AWS S3** (or DigitalOcean Spaces) for storing large media files and model checkpoints  

## Directory Structure



## Getting Started

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your‑username/artrevive.git
   cd artrevive

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver


## Acknowledgments
Developed under the guidance of Esprit School of Engineering faculty.

Thanks to the Hugging Face and TensorFlow communities for model libraries and tutorials.

Special thanks to the ArtRevive team members and open‑source contributors
