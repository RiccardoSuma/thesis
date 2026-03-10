import torch
import numpy as np
import re
import ollama
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from rank_bm25 import BM25Okapi
from modules.vectorization.nomic_wrapper import NomicEmbedder

class InfoRetriever:
    def __init__(self, db_wrapper, device=None):
        self.db = db_wrapper
        # Inizializzo Nomic invece di CLIP
        self.embedder = NomicEmbedder(device=device)
        self._translation_cache = {}  # Cache per risparmiare chiamate a Ollama

    def _enrich_audio_context(self, payload, window_seconds=15.0):

        """
        Prende un payload audio, interroga Qdrant per i chunk adiacenti (+/- window_seconds)
        nello stesso video, e li unisce in un unico testo contestualizzato.
        """
        source_video = payload.get("source")
        base_time = payload.get("timestamp")
        
        # Se non è audio o mancano metadati, restituisce il testo originale
        if payload.get("modality") != "audio" or not source_video or base_time is None:
            return payload.get("text", "")

        # Creo il filtro temporale per Qdrant
        # Stesso video, stesso tipo (audio), timestamp nel range [base - 15, base + 15]
        time_filter = Filter(
            must=[
                FieldCondition(key="source", match=MatchValue(value=source_video)),
                FieldCondition(key="modality", match=MatchValue(value="audio")),
                FieldCondition(
                    key="timestamp", 
                    range=Range(
                        gte=max(0.0, base_time - window_seconds), # Non scendere sotto lo zero
                        lte=base_time + window_seconds
                    )
                )
            ]
        )



        records, _ = self.db.scroll(
            scroll_filter=time_filter,
            limit=100 # 50 chunk sono circa 30-40 secondi di audio
        )

        if not records:
            return payload.get("text", "")

        # Ordino i chunk estratti per timestamp per ricostruire il discorso cronologico
        sorted_records = sorted(records, key=lambda x: x.payload.get("timestamp", 0))

        # Unisco i testi
        enriched_text = " ".join([r.payload.get("text", "").strip() for r in sorted_records])
        
        # Aggiungo un tag per il debugging/LLM per fargli capire che è un blocco esteso
        start_t = sorted_records[0].payload.get("timestamp", base_time)
        end_t = sorted_records[-1].payload.get("timestamp", base_time)
        
        final_content = f"[Trascrizione continua da {start_t}s a {end_t}s]:\n{enriched_text}"
        return final_content

    def _encode(self, text):
        """
        Genera l'embedding della query usando Nomic.
        Il wrapper aggiunge automaticamente il prefisso 'search_query: '.
        """
        return self.embedder.embed_query(text)

    def _tokenize(self, text):
        return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()).split()

    def _translate_to_english(self, text):
        """
        Traduce la query in Inglese per il retrieval visivo (Qwen descrive in EN).
        Mantiene la cache per velocità.
        """
        key = text.strip()
        if key in self._translation_cache:
            return self._translation_cache[key]

        # Se la query è già molto breve o sembra codice, evito Ollama
        if len(key.split()) < 3 and key.isascii():
             return key

        try:
            # Usiamo un modello piccolo per la traduzione rapida
            response = ollama.chat(
                model="deepseek-r1:14b", # o "llama3" o "gemma"
                messages=[{
                    "role": "user",
                    "content": (
                        'Translate this technical query from Italian to English. '
                        'Output ONLY the translated text, no explanation:\n'
                        f'"{text}"'
                    )
                }]
            )
            translated = response["message"]["content"].strip().replace('"', "").replace("'", "").strip()
            # Fallback se il modello impazzisce e restituisce vuoto
            if not translated:
                translated = text
        except Exception as e:
            print(f"⚠️ Translation failed: {e}")
            translated = text

        self._translation_cache[key] = translated
        return translated

    def _normalize_vector(self, vec):
        v = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(v) + 1e-6
        return v / norm


    def _mmr_selection(self, candidates, top_k, lambda_param=0.85): 
        """
        Maximal Marginal Relevance (MMR) per diversificare i risultati.
        lambda_param alto (0.85) privilegia la rilevanza, ma scarta duplicati esatti.
        """
        if not candidates:
            return []

        # Ordina per score ibrido iniziale
        candidates.sort(key=lambda x: x["h_score"], reverse=True)
        
        # Se non ho vettori (caso raro), ritorno i top_k grezzi
        if any(c.get("vector_norm") is None for c in candidates):
            return candidates[:top_k]

        selected = [candidates[0]]
        pool = candidates[1:]

        while len(selected) < top_k and pool:
            # Matrice dei vettori già selezionati
            selected_vecs = np.stack([s["vector_norm"] for s in selected], axis=0)

            best_idx = -1
            best_mmr_score = -1e18

            for i, cand in enumerate(pool):
                rel = float(cand["h_score"])
                
                # Calcola similarità con tutti i selezionati e prendi la massima
                sims = selected_vecs @ cand["vector_norm"]
                max_sim = float(np.max(sims)) if sims.size else 0.0

                # Formula MMR: Rilevanza - (1-lambda)*Similarità
                mmr = (lambda_param * rel) - ((1.0 - lambda_param) * max_sim)
                
                if mmr > best_mmr_score:
                    best_mmr_score = mmr
                    best_idx = i

            if best_idx == -1:
                break
            selected.append(pool.pop(best_idx))

        return selected

    def _rerank_visual(self, query_text, candidates, weight_vector=0.75, weight_bm25=0.25):
        """
        Reranking Ibrido: Nomic Vector Score + BM25 Keyword Score.
        """
        if not candidates:
            return []

        # 1. Dedup veloce basato su contenuto
        seen = set()
        deduped = []
        for c in candidates:
            h = hash(c["content"][:100]) # Hash dei primi 100 caratteri
            if h not in seen:
                seen.add(h)
                deduped.append(c)
        candidates = deduped

        # 2. Normalizzazione Score Vettoriale (Z-Score robusto)
        raw_scores = np.array([float(c["point"].score) for c in candidates], dtype=np.float32)
        if len(raw_scores) > 1 and raw_scores.std() > 0:
            # Standardizzazione
            z = (raw_scores - raw_scores.mean()) / raw_scores.std()
            # Sigmoide per riportare tra 0 e 1
            norm_scores = (1.0 / (1.0 + np.exp(-z))).tolist()
        else:
            norm_scores = [0.8] * len(raw_scores) # Fallback piatto

        # 3. Calcolo BM25 (Keyword Matching)
        corpus = [c["content"] for c in candidates]
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        
        bm25_norm = [0.0] * len(candidates)
        if tokenized_corpus:
            try:
                bm25 = BM25Okapi(tokenized_corpus)
                q_tokens = self._tokenize(query_text)
                bm25_raw = bm25.get_scores(q_tokens)
                
                max_bm25 = float(max(bm25_raw))
                if max_bm25 > 0:
                    bm25_norm = [float(s) / max_bm25 for s in bm25_raw]
            except Exception:
                pass # Se BM25 fallisce, uso solo vettori

        processed = []
        for i, cand in enumerate(candidates):
            v_score = float(norm_scores[i])
            k_score = float(bm25_norm[i])
            
            
            # Score Ibrido Finale
            hybrid = ((v_score * weight_vector) + (k_score * weight_bm25))

            # Preparazione vettore per MMR
            raw_vec = getattr(cand["point"], "vector", None)
            vec_norm = self._normalize_vector(raw_vec) if raw_vec is not None else None

            processed.append({
                "point": cand["point"],
                "content": cand["content"],
                "h_score": float(hybrid),
                "vector_norm": vec_norm,
            })

        return processed
    
    def gt_eval_ids(self, query_text: str, top_k: int) -> list:
        """
        Metodo per estrarre solo gli ID ordinati per rilevanza,
        ottimizzato per il calcolo di nDCG, MRR e Recall.
        """
        english_query = self._translate_to_english(query_text)
        vector_visual = self._encode(english_query)
        vector_audio = self._encode(english_query)

        combined_results = []
        seen_ids = set()

        # --- A. RICERCA VISUALE ---
        visual_candidates = self.db.search(
            query_vector=vector_visual,
            top_k=30, 
            query_filter=Filter(must=[FieldCondition(key="modality", match=MatchValue(value="visual"))]),
            with_vectors=True
        )

        visual_pool = [{"point": p, "content": p.payload.get("text", "")} for p in visual_candidates if len(p.payload.get("text", "")) > 10]
        scored_visuals = self._rerank_visual(english_query, visual_pool)
        selected_visuals = self._mmr_selection(scored_visuals, top_k=4)

        for item in selected_visuals:
            point = item["point"]
            combined_results.append({
                "id": str(point.id),
                "score": float(item.get("h_score", point.score)) # Uso lo score del reranker
            })
            seen_ids.add(point.id)

        # --- B. RICERCA AUDIO ---
        slots_left = top_k - len(combined_results) + 3
        if slots_left > 0:
            audio_candidates = self.db.search(
                query_vector=vector_audio, 
                top_k=slots_left,
                query_filter=Filter(must=[FieldCondition(key="modality", match=MatchValue(value="audio"))]),
                with_vectors=False
            )

            audio_pool = [{"point": p, "content": p.payload.get("text", "")} for p in audio_candidates if p.id not in seen_ids and len(p.payload.get("text", "")) > 10]
            
            scored_audio = self._rerank_visual(english_query, audio_pool) 
            selected_audio = self._mmr_selection(scored_audio, top_k=slots_left)

            for item in selected_audio:
                point = item["point"]
                combined_results.append({
                    "id": str(point.id),
                    "score": float(item.get("h_score", point.score))
                })


        combined_results.sort(key=lambda x: x["score"], reverse=True)

        # Restituisco solo la lista piatta degli ID testuali, tagliata al top_k richiesto
        return [res["id"] for res in combined_results][:top_k]

    def get_answer(self, query_text: str, top_k: int):
        print(f"\n🔍 Processing Query: {query_text}")
        
        # 1. Traduzione per ricerca visiva (Qwen è in Inglese)
        english_query = self._translate_to_english(query_text)
        print(f"   🇬🇧 Translated: {english_query}")

        # 2. Embedding 
        vector_visual = self._encode(english_query)
        vector_audio = self._encode(english_query)

        final_context = []
        seen_ids = set()

        # --- A. RICERCA VISUALE (SLIDE INTELLIGENTI) ---
        # Cerco di più inizialmente per poi filtrare con Rerank/MMR
        visual_candidates = self.db.search(
            query_vector=vector_visual,
            top_k=30, 
            query_filter=Filter(must=[FieldCondition(key="modality", match=MatchValue(value="visual"))]),
            with_vectors=True # Serve per MMR
        )

        visual_pool = []
        for p in visual_candidates:
            # Ora uso 'text' che contiene la descrizione ricca di Qwen
            content = p.payload.get("text", "")
            if len(content) > 10:
                visual_pool.append({"point": p, "content": content})

        # Rerank & MMR
        scored_visuals = self._rerank_visual(english_query, visual_pool)
        selected_visuals = self._mmr_selection(scored_visuals, top_k=4) # Voglio max 4 slide forti

        for item in selected_visuals:
            p = item["point"].payload
            final_context.append({
                "source": p.get("source"),
                "timestamp": p.get("timestamp"),
                "modality": "VISUAL",
                "score": float(item["h_score"]),
                "content_to_use": item["content"], # Descrizione Qwen
                "image_path": p.get("image_path")  # Path per la UI
            })
            seen_ids.add(item["point"].id)

        # --- B. RICERCA AUDIO (TRASCRIZIONI) ---
        # Riempio lo spazio rimanente con l'audio
        slots_left = top_k - len(final_context) + 3 # +2 di bonus
        
        if slots_left > 0:
            audio_candidates = self.db.search(
                query_vector=vector_audio, 
                top_k=slots_left,
                query_filter=Filter(must=[FieldCondition(key="modality", match=MatchValue(value="audio"))]),
                with_vectors=False
            )

            audio_pool = []
            for anchor in audio_candidates:
                if anchor.id in seen_ids: continue
                text = anchor.payload.get("text", "")
                if len(text) > 10:
                    audio_pool.append({"point": anchor, "content": text})

            
            scored_audio = self._rerank_visual(english_query, audio_pool)
            
            selected_audio = self._mmr_selection(scored_audio, top_k=slots_left)

            # 5. Context Expansion solo sui "Vincitori"
            for item in selected_audio:
                anchor = item["point"]
                p = anchor.payload
                
                # Chiamo la funzione che interroga Qdrant per prendere +/- 15 secondi.
                expanded_text = self._enrich_audio_context(p, window_seconds=30.0)
                
                # Controllo di sicurezza
                if len(expanded_text) > 15:
                    final_context.append({
                        "source": p.get("source"),
                        "timestamp": p.get("timestamp"), # Timestamp del chunk "ancora"
                        "modality": "AUDIO",
                        "score": float(anchor.score),
                        "content_to_use": expanded_text, # Passiamo il testo espanso al LLM!
                    })
                    seen_ids.add(anchor.id)

        # Ordino per timestamp per coerenza narrativa (aiuta il LLM a leggere in ordine cronologico)
        return sorted(final_context, key=lambda x: (x["source"], x["timestamp"]))