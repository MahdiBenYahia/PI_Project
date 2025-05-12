#%% Importations consolidées
from datetime import time
import logging
from bs4 import BeautifulSoup
import requests
import wikipedia
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    ViTModel,
    ViTImageProcessor,
    TrainingArguments,
    Trainer
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.embeddings import HuggingFaceEmbeddings
import faiss
import numpy as np
import torch
import json
from typing import List
from datasets import Dataset
import traceback
import gc
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import os
from pathlib import Path

BASE_DIR_ART_FILE = Path(__file__).resolve().parent.parent

def cleanup_memory():
    torch.cuda.empty_cache()
    gc.collect()

#%% Partie 1 - Crawler Multi-Sources avec Structuration
class EnhancedArtDataCrawler:
    def __init__(self):
        self.sources = {
            'wikipedia': self._crawl_wikipedia,
            'met_museum': self._crawl_metmuseum,
            'art_encyclopedia': self._crawl_art_encyclopedia,
            'wikiart': self._crawl_wikiart,
            'arthistory': self._crawl_arthistory
        }

    def _crawl_wikipedia(self, artist) -> List[Document]:
        try:
            wikipedia.set_lang("en")
            page = wikipedia.page(artist, auto_suggest=False)
            # Nouveau contenu enrichi
            full_content = f"""
            Artiste: {artist}
            Biographie: {page.summary}
            Mouvements Artistiques: {self._get_artist_movements(page.content)}
            Œuvres Majeures: {self._get_major_works(page.content)}
            Héritage: {self._extract_section(page.content, 'Legacy')}
            Techniques: {self._extract_techniques(page.content)}
            """
            return [Document(
                page_content=full_content,
                metadata={
                    "source": page.url,
                    "categories": page.categories,
                    "last_modified": page.revision_date
                }
            )]
        except:
            return []


    def _extract_section(self, content, section_title):
        """Extrait le contenu d'une section spécifique"""
        start_idx = content.find(f"== {section_title} ==")
        if start_idx == -1: return ""
        end_idx = content.find("==", start_idx + 4)
        return content[start_idx:end_idx].strip()



    def _get_artist_movements(self, content):
        """Extrait les mouvements artistiques"""
        movements = []
        if "associated with" in content:
            movements = content.split("associated with")[1].split(".")[0].strip()
        return movements



    def _crawl_metmuseum(self, artist) -> List[Document]:
        try:
            response = requests.get(f"https://collectionapi.metmuseum.org/public/collection/v1/search?q={artist}", timeout=25)
            data = response.json()
            object_ids = data.get('objectIDs', [])[:20]

            documents = []
            for obj_id in object_ids[:20]:  # Limite à 3 œuvres
                obj_response = requests.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}", timeout=25)
                obj_data = obj_response.json()
                if obj_data.get('artistDisplayName', '').lower() != artist.lower():
                      continue  # Skip si l'artiste ne correspond pas
                content = f"Title: {obj_data.get('title','')}\nMedium: {obj_data.get('medium','')}\nDate: {obj_data.get('objectDate','')}"
                documents.append(Document(
                      page_content=content,
                      metadata={
                          "source": f"MetObjectID:{obj_id}",
                          "department": obj_data.get('department', ''),
                          "artist": artist,
                          "period": obj_data.get('period', ''),
                          "culture": obj_data.get('culture', ''),
                          "classification": obj_data.get('classification', '')
                      }
                    ))
            return documents
        except:
            return []

    def _crawl_art_encyclopedia(self, artist) -> List[Document]:
        try:
            # Nouveau parsing enrichi
            formatted_artist = artist.replace('_', '+')
            url = f"http://www.artcyclopedia.com/artists/{formatted_artist}.html"
            response = requests.get(url, timeout=25)
            soup = BeautifulSoup(response.text, 'html.parser')

            full_content = []
            # Ajout de nouvelles sections
            biography = soup.find('h2', text='Biography')
            if biography:
                full_content.append("Biographie: " + biography.find_next('p').get_text(strip=True))

            movements = soup.find('b', text='Style:')
            if movements:
                full_content.append("Mouvements: " + movements.next_sibling.strip())

            influences = soup.find('b', text='Influences:')
            if influences:
                full_content.append("Influences: " + influences.next_sibling.strip())

            return [Document(
                page_content="\n".join(full_content),
                metadata={"source": url}
            )] if full_content else []
        except:
            return []


    def _crawl_wikiart(self, artist) -> List[Document]:
        """Nouveau crawler pour WikiArt.org"""
        try:
            url = f"https://www.wikiart.org/en/{artist.replace(' ', '-').lower()}"
            response = requests.get(url, timeout=25)
            soup = BeautifulSoup(response.text, 'html.parser')

            content = []
            # Extraction des informations stylistiques
            style_info = soup.find('ul', class_='dictionaries-list')
            if style_info:
                content.append("Style: " + style_info.get_text(separator=" | "))

            # Extraction des analyses d'œuvres
            artworks = soup.find_all('article', class_='painting-row')[:20]
            for artwork in artworks:
                title = artwork.find('h3').get_text(strip=True)
                analysis = artwork.find('div', class_='description').get_text(strip=True)
                content.append(f"Œuvre: {title}\nAnalyse: {analysis}")

            return [Document(
                page_content="\n".join(content),
                metadata={"source": url}
            )] if content else []
        except:
            return []



    def _crawl_arthistory(self, artist) -> List[Document]:
        """Nouveau crawler pour arthistory.net"""
        try:
            url = f"https://www.arthistory.net/{artist.replace(' ', '').lower()}/"
            response = requests.get(url, timeout=25)
            soup = BeautifulSoup(response.text, 'html.parser')

            content = []
            # Extraction des caractéristiques stylistiques
            style_section = soup.find('h3', text='Artistic Style')
            if style_section:
                content.append(style_section.find_next('p').get_text(strip=True))

            # Extraction des influences historiques
            influences = soup.find('h3', text='Historical Context')
            if influences:
                content.append(influences.find_next('ul').get_text(separator="\n"))

            return [Document(
                page_content="\n".join(content),
                metadata={"source": url}
            )] if content else []
        except:
            return []

    def crawl(self, artists: List[str], max_docs=100) -> List[Document]:
        # Optimisation pour les requêtes parallèles
        with ThreadPoolExecutor(max_workers=10) as executor:  # Augmenter les workers
            futures = []
            for artist in artists:
                for source_name, source in self.sources.items():
                    futures.append(executor.submit(
                        self._safe_crawl, 
                        source, 
                        artist,
                        source_name
                    ))

            documents = []
            for future in as_completed(futures):
                try:
                    docs = future.result()
                    documents.extend(docs)
                    if len(documents) >= max_docs:
                        break
                except Exception as e:
                    print(f"Erreur: {str(e)}")
        return documents[:max_docs]

        # Déduplication restante...
        return documents[:max_docs]

    def _safe_crawl(self, source_func, artist, source_name):
        """Execute un crawler avec gestion d'erreur détaillée"""
        try:
            start_time = time.time()
            result = source_func(artist)
            elapsed = time.time() - start_time
            print(f"{source_name}: {len(result)} documents ({elapsed:.2f}s)")
            return result
        except Exception as e:
            error_msg = f"{source_name} - {artist} erreur: {str(e)}"
            print(error_msg)
            logging.error(error_msg)
            return []
        except KeyboardInterrupt:
            return []

#%% Partie 2 - RAG Avancé avec Validation
class EnhancedRAG:
    def __init__(self, documents: List[Document]):
        self.vectorstore = self._init_vectorstore(documents)
        self._validate_content()

    def _init_vectorstore(self, documents: List[Document]) -> FAISS:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,  # Augmenté de 512 à 1024
            chunk_overlap=256,
            separators=["\n\n## ", "\n\n", "\n", ". ", " ", ""],  # Nouveaux séparateurs
            keep_separator=True
        )
        chunks = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",  # Modèle plus puissant
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return FAISS.from_documents(chunks, embeddings)

    def _validate_content(self):
        try:
            test_query = "art movement"
            results = self.vectorstore.similarity_search(test_query, k=1)
            if not results or len(results[0].page_content) < 20:
                raise ValueError("Contenu RAG non valide")
        except Exception as e:
            print(f"Validation RAG échouée: {str(e)}")
            traceback.print_exc()

    def visualize_embeddings(self):
        try:
            embeddings = self.vectorstore.index.reconstruct_n(0, self.vectorstore.index.ntotal)
            print(f"Dimensions des embeddings: {embeddings.shape}")
            print(f"Distribution: μ={np.mean(embeddings):.2f}, σ={np.std(embeddings):.2f}")
        except Exception as e:
            print(f"Visualisation impossible: {str(e)}")


class LoRALLM:
    def __init__(self, model_name="microsoft/phi-2", lora_r=16, device="cuda"):
        # Chargement du modèle et tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )

        # Ajouter la configuration du pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Utiliser eos_token comme pad_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=2*lora_r,
            target_modules=[
                "q_proj",  # Projection des queries
                "k_proj",  # Projection des keys
                "v_proj",  # Projection des values
                "fc1",     # Première couche feed-forward
                "fc2"      # Deuxième couche feed-forward
            ],
            task_type="CAUSAL_LM",
            bias="none",
            lora_dropout=0.05
        )

        # Application de LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.to(device)

        # Nouvelle configuration de génération
        self.generation_config = {
            "max_new_tokens": 150,  # Réduit de 256
            "temperature": 0.5,     # Plus déterministe
            "top_p": 0.85,
            "do_sample": True,
            "num_beams": 2         # Acceleration de la génération
        }

    def train_model(self, dataset, epochs=3):
        """Entraînement avec gestion des formats"""
        # Formatage des données
        def format_fn(examples):
            return self.tokenizer(
                [q + " " + a for q, a in zip(examples["question"], examples["answer"])],
                padding="max_length",
                truncation=True,
                max_length=512
            )

        dataset = dataset.map(format_fn, batched=True)

        # Paramètres d'entraînement
        training_args = TrainingArguments(
            output_dir="./phi2-lora",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            num_train_epochs=epochs,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none"  # Désactivation W&B
        )

        # Entraîneur personnalisé
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=lambda data: {
                "input_ids": torch.stack([torch.tensor(d["input_ids"]) for d in data]),
                "attention_mask": torch.stack([torch.tensor(d["attention_mask"]) for d in data]),
                "labels": torch.stack([torch.tensor(d["input_ids"]) for d in data])
            }
        )

        trainer.train()

    def save_model(self, output_dir="./fine_tuned_model"):
        """Sauvegarde du modèle et du tokenizer"""
        # Création du répertoire si nécessaire
        os.makedirs(output_dir, exist_ok=True)

        # Sauvegarde du modèle LoRA
        self.model.save_pretrained(output_dir)

        # Sauvegarde du tokenizer
        self.tokenizer.save_pretrained(output_dir)

        # Sauvegarde de la configuration de génération
        with open(os.path.join(output_dir, "generation_config.json"), "w") as f:
            json.dump(self.generation_config, f)

        print(f"Modèle et tokenizer sauvegardés dans {output_dir}")

    @classmethod
    def from_pretrained(cls, model_path, device="cuda"):
        """Chargement du modèle pré-entraîné avec LoRA"""
        # Création d'une instance vide
        instance = cls.__new__(cls)

        # Chargement du tokenizer
        instance.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Chargement du modèle avec LoRA
        instance.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Chargement de la configuration de génération si elle existe
        gen_config_path = os.path.join(model_path, "generation_config.json")
        if os.path.exists(gen_config_path):
            with open(gen_config_path, "r") as f:
                instance.generation_config = json.load(f)
        else:
            # Configuration par défaut
            instance.generation_config = {
                "max_new_tokens": 150,
                "temperature": 0.5,
                "top_p": 0.85,
                "do_sample": True,
                "num_beams": 2
            }

        # Déplacement du modèle sur le bon appareil
        instance.model.to(device)

        print(f"Modèle chargé depuis {model_path}")
        return instance

    def generate(self, prompt, **kwargs):
        # Nouveau format de prompt avec délimiteurs clairs
        formatted_prompt = f"""<|im_start|>user
        {prompt}
        <|im_end|>
        <|im_start|>assistant
        """

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(self.model.device)
        # Ajout de stop tokens pour limiter la génération
        stop_token_ids = [
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.tokenizer.eos_token_id
        ]
        outputs = self.model.generate(
            **inputs,
            **self.generation_config,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=stop_token_ids,
            no_repeat_ngram_size=3  # Réduction des répétitions
        )
        # Décodage avec nettoyage
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_response.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()

#%% Partie 4 - Pipeline Intégré
class ArtAnalysisPipeline:
    def __init__(self, artist_file=None, device="cuda"):  # Modifier le fichier par défaut
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Chemin par défaut si non spécifié
        self.artist_file = artist_file or os.path.join(current_dir, "peintre.txt")
        # Vérification de l'existence du fichier
        if not os.path.exists(self.artist_file):
            raise FileNotFoundError(f"Fichier d'artistes introuvable : {self.artist_file}")
        self.device = device
        self.initialized = False  # Nouveau flag d'initialisation

    def async_initialize(self):
        """Initialisation asynchrone avec crawling"""
        if not self.initialized:
            with open(self.artist_file, 'r', encoding='utf-8') as f:
                artists = [line.strip() for line in f if line.strip()]
            print(f"Chargement de {len(artists)} artistes depuis {self.artist_file}")
            
            self.crawler = EnhancedArtDataCrawler()
            documents = self.crawler.crawl(artists)
            self.rag = EnhancedRAG(documents)
            self.llm = LoRALLM(device=self.device)
            self.initialized = True
            print("Crawl initial terminé avec", len(documents), "documents")

    def _refresh_rag(self, artists, max_docs=200):
        """Met à jour le RAG avec de nouvelles données"""
        new_docs = self.crawler.crawl(artists, max_docs)
        
        if hasattr(self, 'rag'):
            # Mise à jour incrémentielle
            existing_docs = self.rag.vectorstore.docstore._dict.values()
            all_docs = list(existing_docs) + new_docs
            self.rag.vectorstore = FAISS.from_documents(
                all_docs, 
                self.rag.vectorstore.embeddings
            )
        else:
            # Première initialisation
            self.rag = EnhancedRAG(new_docs)
        
        print(f"RAG mis à jour avec {len(new_docs)} nouveaux documents")

    def dynamic_crawl(self, query):
        """Déclenche un crawl ciblé basé sur la requête"""
        # Extraction des entités nommées
        artists = self._extract_artists(query)
        
        if artists:
            print(f"Lancement d'un crawl ciblé pour {artists}")
            self._refresh_rag(artists)
            return True
        return False

    def _extract_artists(self, text):
        """Extrait les noms d'artistes avec une regex améliorée"""
        pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b"
        matches = re.findall(pattern, text)
        return list(set(matches))[:3]  # Limite à 3 artistes max


    def load_llm_model(self):
        if not self._llm_loaded:
            self.llm = LoRALLM(device=self.device)
            self._llm_loaded = True

    def unload_models(self):
        #del self.vision_encoder
        del self.llm
        cleanup_memory()
        #self._vision_loaded = False
        self._llm_loaded = False

    def prepare_training_data(self, qa_pairs):
        return Dataset.from_dict({
            "question": [pair[0] for pair in qa_pairs],
            "answer": [pair[1] for pair in qa_pairs]
        })

    def finetune_models(self, vision_data=None, llm_data=None, epochs=3):
        if llm_data:
            self.llm.train_model(llm_data, epochs)


    def auto_generate_training_data(self, num_samples=100):
        # Convertir les valeurs du dictionnaire en liste
        all_docs = list(self.rag.vectorstore.docstore._dict.values())

        # Vérifier qu'on a assez de documents
        if len(all_docs) < num_samples:
            print(f"Warning: Only {len(all_docs)} documents available, using all")
            num_samples = len(all_docs)

        documents = random.sample(all_docs, min(num_samples, len(all_docs)))

        qa_pairs = []
        for doc in documents:
            content = doc.page_content
            artist = doc.metadata.get('artist', 'unknown')

            # Génère des questions avec le LLM
            prompt = f"""GENERATE ONLY 1 QUESTION-ANSWER PAIR. STRICT FORMAT:
            Context: {content[:800]}
            Question:[question here?]
            Answer:[answer here]
            NO ADDITIONAL TEXT!"""
            generated = self.llm.generate(prompt)

            for pair in generated.split('\n'):
                if '|' in pair:
                    q, a = pair.split('|', 1)
                    qa_pairs.append((q.strip(), a.strip()))

        return self.prepare_training_data(qa_pairs[:num_samples])


    def save(self, output_dir="./art_pipeline"):
        """Sauvegarde complète du pipeline"""
        os.makedirs(output_dir, exist_ok=True)

        # Sauvegarde du LLM
        llm_dir = os.path.join(output_dir, "llm")
        self.llm.save_model(llm_dir)

        # Sauvegarde du RAG (FAISS)
        faiss_index_path = os.path.join(output_dir, "faiss_index")
        self.rag.vectorstore.save_local(faiss_index_path)

        print(f"Pipeline sauvegardé dans {output_dir}")

    @classmethod
    def load(cls, input_dir="./art_pipeline", device="cuda"):
        """Chargement complet du pipeline"""
        # Création d'une instance vide
        instance = cls.__new__(cls)
        instance.device = device

        # Chargement du LLM
        llm_dir = os.path.join(input_dir, "llm")
        instance.llm = LoRALLM.from_pretrained(llm_dir, device=device)
        instance._llm_loaded = True

        # Chargement du RAG
        faiss_index_path = os.path.join(input_dir, "faiss_index")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        instance.rag = EnhancedRAG.__new__(EnhancedRAG)
        instance.rag.vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

        print(f"Pipeline chargé depuis {input_dir}")
        return instance



#%% Partie 5 - Évaluation Avancée
class ArtEvaluator:
    def __init__(self, pipeline):
          self.pipeline = pipeline


    def interactive_evaluation(self, test_image=None):
        metrics = {}

        if test_image:
            original = self.pipeline.vision_encoder.encode_image(test_image)
            augmented = original + torch.randn_like(original)*0.1
            metrics["vision_similarity"] = torch.cosine_similarity(original, augmented).mean().item()

        # Mode interactif RAG+LLM
        while True:
            query = input("\nEnter your art-related query (or 'exit'): ")
            if query.lower() == 'exit':
                break

            # Recherche RAG
            docs = self.pipeline.rag.vectorstore.similarity_search(
                query,
                k=3,
                #filter=lambda doc: artist.lower() in doc.metadata.get('artist', '').lower()
            )
            print("\nTop 3 RAG Results:")
            for i, doc in enumerate(docs, 1):
                print(f"{i}. Source: {doc.metadata['source']}")
                print(f"Content: {doc.page_content[:200]}...\n")

            # Génération et évaluation automatique
            response = self.pipeline.llm.generate(query)
            docs = self.pipeline.rag.vectorstore.similarity_search(query, k=3)
            accuracy = self._dynamic_accuracy_assessment(query, response, docs)

            print(f"\nLLM Response: {response}")
            print(f"\nAssessed Accuracy: {accuracy:.2f}")

        return metrics

    def _dynamic_accuracy_assessment(self, query, response, docs):
        # Nouveau prompt structuré
        verification_prompt = f"""ANALYZE STRICTLY BASED ON CONTEXT. RESPONSE FORMAT: 'Score: 0.XX'

        Query: {query}
        Response: {response}
        Context: {'###'.join([d.page_content for d in docs])}

        Evaluation criteria:
        1. Relevance to query (0-0.4)
        2. Context alignment (0-0.4)
        3. Factual correctness (0-0.2)

        Score: """

        try:
            raw_output = self.pipeline.llm.generate(verification_prompt)
            # Nouvelle extraction robuste
            score_match = re.search(r'\b0?\.\d{1,2}\b', raw_output)
            return float(score_match.group()) if score_match else 0.0
        except:
            return 0.0

if __name__ == "__main__":
    cleanup_memory()

    # Option 1: Créer et entraîner un nouveau pipeline
    if not os.path.exists("./art_pipeline"):
        print("Création d'un nouveau pipeline...")
        pipeline = ArtAnalysisPipeline(artist_file="/content/peintre.txt")
        pipeline.load_llm_model()

        # Génération auto de données
        train_data = pipeline.auto_generate_training_data(num_samples=50)
        pipeline.finetune_models(llm_data=train_data, epochs=5)

        # Sauvegarde du pipeline
        pipeline.save("./art_pipeline")
        
    
    # Option 2: Charger un pipeline existant
    else:
        print("Chargement d'un pipeline existant...")
        pipeline = ArtAnalysisPipeline.load("./art_pipeline")

    # Évaluation interactive
    evaluator = ArtEvaluator(pipeline)
    evaluator.interactive_evaluation()
    cleanup_memory()