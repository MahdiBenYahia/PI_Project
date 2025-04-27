from django.shortcuts import render
import tensorflow as tf
from keras.models import load_model
from django.conf import settings
from django.http import JsonResponse
import numpy as np
from PIL import Image
import io
from . import forms
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


from django.core.files.storage import FileSystemStorage

def predict(request):
    if request.method == 'POST':
        img = request.FILES.get('image')

        if img:
            # Sauvegarde de l'image téléchargée
            fs = FileSystemStorage()
            filename = fs.save(img.name, img)
            uploaded_image_url = fs.url(filename)  # URL de l'image enregistrée

            image = Image.open(img)
            image = image.resize((128, 128))  # Redimensionner à 128x128
            image = np.array(image) / 255.0  # Normaliser
            image = np.expand_dims(image, axis=0)  # Ajouter la dimension du batch

            # Prédiction
            prediction = model.predict(image)
            predicted_class = 'human' if prediction[0][0] > 0.5 else 'AI'

            result = {
                'prediction': predicted_class,  # La classe prédite
                'probability': prediction.tolist(),  # Probabilité
                'image_url': uploaded_image_url  # URL de l'image téléchargée
            }
            return JsonResponse(result)

        return JsonResponse({'error': 'No image uploaded'}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=400)

