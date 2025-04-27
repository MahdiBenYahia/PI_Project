from django.shortcuts import render
import tensorflow as tf
from keras.models import load_model
from django.conf import settings
from django.http import JsonResponse
import numpy as np
from PIL import Image
import io
# Load model globally so it isn't reloaded on every request
model = load_model(settings.MODEL_PATH)

def home(request):
    return render(request, 'base.html')

def classify_view(request):
    return render(request, 'classify.html')

def upload_artwork(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            # handle form.cleaned_data or save
            pass
    else:
        form = UploadImageForm()

    return render(request, 'classify.html', {'form': form})


def predict(request):
    if request.method == 'POST':
        img = request.FILES.get('image')

        if img:
            image = Image.open(img)
            image = image.resize((128, 128))  # Resize to your model's input size
            image = np.array(image) / 255.0  # Normalize if needed
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            prediction = model.predict(image)
            
            # Supposons que la classe 1 est 'human' et la classe 0 est 'AI'
            predicted_class = 'human' if prediction[0][0] > 0.5 else 'AI'

            result = {
                'prediction': predicted_class,  # Renvoi de la classe au lieu de la probabilité brute
                'probability': prediction.tolist()  # Vous pouvez garder la probabilité également
            }
            return JsonResponse(result)

        return JsonResponse({'error': 'No image uploaded'}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=400)
