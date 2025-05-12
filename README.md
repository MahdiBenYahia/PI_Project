# PI_Project
🧱 2D to 3D Statue Reconstruction with Hunyuan3D
This project allows you to convert a 2D image of a statue into a realistic 3D mesh using the Hunyuan3D model by Tencent. The output is exported in .glb format, making it suitable for viewing in 3D environments or importing into AR/VR applications.

🎯 Goal
Transform flat 2D photos of monuments or statues into full 3D objects using a pre-trained AI model — no 3D modeling skills required.

🧠 Model Used
Hunyuan3D-DiT (Diffusion Transformer)
A flow-matching diffusion model capable of high-quality 3D mesh reconstruction from a single image.

📦 Source: tencent/Hunyuan3D-2

🛠️ How It Works
✅ Upload a statue image (.jpg or .png)

✅ Run the notebook (Colab recommended)

✅ The image is processed and reconstructed into a .glb 3D mesh file

✅ Download the result and use it in 3D viewers or engines (Blender, Unity, etc.)

🔧 Setup (in Google Colab)
python
Copy
Edit
# 1. Install the required libraries
!pip install hy3dgen trimesh

# 2. Upload your statue image
from google.colab import files
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# 3. Load the model
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')

# 4. Generate the 3D mesh
mesh = pipeline(image=image_path)[0]

# 5. Export and download
mesh.export('statue_output.glb')
files.download('statue_output.glb')
📂 Output Format
.glb (GL Transmission Format Binary) — compatible with:

Blender
Three.js
UnityWeb 3D viewers
