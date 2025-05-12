# PI_Project

🏛️ Automated Captioning of Monument Images using Gemini 2.0 Flash
This project uses Google's Gemini 2.0 Flash model to generate factual, style-aware descriptions of monument and statue images. The goal is to build a consistent and reliable dataset of text-image pairs for future training or documentation, without altering historical accuracy.

🎯 Objective
To automatically generate short, clear, and factual captions for each monument image, including:

✅ Material (e.g., stone, marble)

✅ Visible damage (e.g., cracks, erosion)

✅ Recognizable artistic style (e.g., Baroque, Gothic)

⚠️ The descriptions are purely observational – no fictional or speculative content is added.

🧠 Model Used
Gemini 2.0 Flash API via google.generativeai

Input: raw image (.jpg, .png)

Output: factual caption (plain text)


🔧 How to Run
1/Mount Google Drive (if using Colab):
from google.colab import drive
drive.mount('/content/drive')
2/Set your API Key:
import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY")
3/Run the notebook to:
-Scan the monument folder
-Generate a description for each image
-Export the results to a .csv file

📍 Example Prompt to Gemini
"Give a short, clear, and factual caption of this monument. Mention the material (stone, marble, etc.), visible damage, and any recognizable artistic style (baroque, gothic, etc.). Don't invent – only describe what is clearly visible."


📦 Output Example:
image :
![110](https://github.com/user-attachments/assets/95629001-23ed-4f8d-935c-9fdb46a4e73b)

decription generated :
This is the Trevi Fountain in Rome, Italy, a large and ornate late Baroque fountain constructed primarily of travertine stone. Some weathering and staining are visible on the stone surfaces.
