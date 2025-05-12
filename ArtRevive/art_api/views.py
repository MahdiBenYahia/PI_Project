from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.http import JsonResponse
from django.views import View
from django.conf import settings
from .pipeline.art_pipeline import ArtAnalysisPipeline, ArtEvaluator
import re
import json
from threading import Thread
from django.core.cache import cache

@method_decorator(csrf_exempt, name='dispatch')
class ArtQueryView(View):
    pipeline = None
    evaluator = None

    @classmethod
    def setup_pipeline(cls):
        if not cls.pipeline:
            cls.pipeline = ArtAnalysisPipeline.load(
                input_dir=settings.ART_PIPELINE_DIR, 
                device=settings.MODEL_DEVICE
            )
            cls.evaluator = ArtEvaluator(cls.pipeline)

    def dispatch(self, request, *args, **kwargs):
        self.setup_pipeline()  # Appel correct de la méthode
        return super().dispatch(request, *args, **kwargs)

    def get(self, request):
        # Déclencher l'initialisation au premier chargement
        pipeline = cache.get('art_pipeline')
        
        if not pipeline:
            pipeline = ArtAnalysisPipeline()  # Utilise peintre.txt par défaut
            Thread(target=pipeline.async_initialize).start()
            cache.set('art_pipeline', pipeline, timeout=None)
        
        return render(request, 'query_form.html')

    def post(self, request):
        query = request.POST.get('query')
        lang = self._detect_language(query)

        # Génération avec prompt adapté
        prompt = self._format_prompt(query, lang)
        raw_response = self.pipeline.llm.generate(prompt)

        # Nettoyage approfondi
        response = self._clean_response(raw_response, lang)
        response = self.clean_response(response)

        # Dernière vérification de redondance
        response = self._remove_duplicate_sentences(response)

        return render(request, 'query_form.html', {'response': response})

    def _detect_language(self, text):
        """Détection simple de la langue"""
        english_words = {'the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'you', 'that'}
        french_words = {'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'où', 'que'}
        
        en_count = sum(1 for word in text.lower().split() if word in english_words)
        fr_count = sum(1 for word in text.lower().split() if word in french_words)
        
        return 'en' if en_count > fr_count else 'fr'

    def _format_prompt(self, query, lang):
        """Génère un prompt adapté à la langue de la question"""
        #lang = self._detect_language(query)
        self.lang = lang

        if lang == 'fr':
            return f"""Réponds de manière claire et structurée en français. 
            Question : {query}
            Réponse concrète :"""
        else:
            return f"""Provide a clear, factual response in English. 
            Question: {query}
            Direct answer:"""
            
    def _clean_response(self, raw_response, lang):
        """Nettoyage avancé de la réponse"""
        # Suppression des patterns de code
        patterns_to_remove = [
            r'def\s+\w+\(.*?\)\s*->\s*\w+:',
            r'return\s+".*?"',
            r'""".*?"""',
            r'\s*\.\.\.\s*'
        ]

        for pattern in patterns_to_remove:
            raw_response = re.sub(pattern, '', raw_response, flags=re.DOTALL)

        # Normalisation des espaces
        cleaned = re.sub(r'\s+', ' ', raw_response).strip()

        # Suppression des phrases incomplètes
        if lang == 'fr':
            cleaned = re.sub(r'\b(et|ou|de)\s*$', '', cleaned)
        else:
            cleaned = re.sub(r'\b(and|or|of)\s*$', '', cleaned)

        return cleaned
    
    
    def clean_response(self, raw_response):
        """Nettoie les artefacts de génération du modèle"""
        # Split sur "Question" ou """
        parts = re.split(r'\nQuestion|\n"""|Question:|"""', raw_response)
        if len(parts) > 1:
            return parts[0].strip()
        return raw_response.strip()
    
    def _remove_duplicate_sentences(self, text):
        """Supprime les phrases identiques consécutives"""
        sentences = re.split(r'(?<=[.!?]) +', text)
        cleaned = []
        prev_sentence = ""

        for sentence in sentences:
            # Comparaison normalisée (minuscules, ponctuation)
            current = re.sub(r'[^\w\s]', '', sentence).lower().strip()
            previous = re.sub(r'[^\w\s]', '', prev_sentence).lower().strip()

            if current != previous:
                cleaned.append(sentence)
                prev_sentence = sentence

        return ' '.join(cleaned)
    
    
    
class CheckInitView(View):
    def get(self, request):
        pipeline = cache.get('art_pipeline')
        return JsonResponse({
            'initialized': pipeline.initialized if pipeline else False
        })