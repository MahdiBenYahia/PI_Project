# PI_Project

🗿 Fine-Tuning Stable Diffusion 1.5 with LoRA for Statue Image Generation
This project demonstrates how to fine-tune Stable Diffusion 1.5 using LoRA (Low-Rank Adaptation) on a custom dataset of statue images and textual prompts. The goal is to teach the model to generate new, high-quality statue visuals based on short, descriptive prompts.

🎯 Objective
To enable text-to-image generation of artistic and historical statues by training a lightweight LoRA adapter on a curated image/prompt dataset. This allows for faster inference and style control without altering the full model weights.
🗂️ Dataset Structure
The training data includes:

📸 Images of statues (resized to 512x512 resolution)

📝 Text descriptions or prompts for each image
Organized as:
PI_Project/
├── data_diffusion/
│   └── data_prep/
│       ├── images/             
│       └── descriptions/      
Each image has a corresponding .txt file with a prompt used for fine-tuning.

🧪 Methodology
Model: Stable Diffusion 1.5

Training Method: LoRA fine-tuning (lightweight and GPU-efficient)

Prompt Generation: Descriptions either written manually or auto-generated

Tools Used: HuggingFace Datasets, Torchvision, Pandas, PIL, Diffusers

⚙️ Setup & Dependencies
Install the required libraries:
pip install torch torchvision diffusers datasets transformers Pillow pandas

🚀 How to Run
1/Clone this repo:
git clone https://github.com/your-username/statue-lora-finetune.git
cd statue-lora-finetune
2/Open and run the Jupyter notebook:
jupyter notebook updated_finetune_notebook.ipynb
3/Update paths to point to your images/ and descriptions/ folders.

4/Run the notebook to:
-Load & preprocess the dataset
-Apply LoRA-based fine-tuning
-Generate new statue images from prompts


🖼️ Sample Output


prompt :A portrait of Hannibal Barca, the Carthaginian general, in ancient military armor, standing on a rocky cliff overlooking the Alps, with war elephants and Carthaginian soldiers in the background ,historical realism, cinematic style"

result:


![Screenshot hannibal](https://github.com/user-attachments/assets/bfab2fba-8b9d-4552-a485-1b3e25ccd038)

