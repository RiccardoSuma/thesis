import torch
import clip
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from rank_bm25 import BM25Okapi
import re
import numpy as np
import ollama

class InfoRetriever:
    def __init__(self, db_wrapper, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.db = db_wrapper
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self._translation_cache = {}  # ✅ Cache per non chiamare Ollama inutilmente

    def _encode(self, text):
        text_input = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0].tolist()

    def _tokenize(self, text):
        return re.sub(r"[^a-zA-Z0-9\s]", "", text.lower()).split()

    def _translate_to_english(self, text):
        """Traduzione con Cache: veloce e risparmia risorse."""
        key = text.strip()
        if key in self._translation_cache:
            return self._translation_cache[key]

        try:
            response = ollama.chat(
                model="mistral",
                messages=[{
                    "role": "user",
                    "content": (
                        'Translate this technical Deep Learning query from Italian to English. '
                        'Output ONLY the translated text:\n'
                        f'"{text}"'
                    )
                }]
            )
            translated = response["message"]["content"].strip().replace('"', "").replace("'", "").strip()
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

    def _is_sticky_content(self, text):
        text_lower = text.lower()
        if "edoardoragusa" in text_lower or "github.com" in text_lower:
            return True
        if text_lower.count("import ") > 4 and text_lower.count(" from ") > 4:
            return True
        return False

    def _mmr_selection(self, candidates, top_k, lambda_param=0.9): 
        """
        TUNING GEMINI: lambda_param = 0.9 (invece di 0.8).
        Tolleriamo slide simili (es. parte 1 e parte 2 della spiegazione).
        """
        if not candidates:
            return []

        candidates.sort(key=lambda x: x["h_score"], reverse=True)
        if len(candidates) <= top_k:
            return candidates

        # Fallback di sicurezza se mancano i vettori
        if any(c.get("vector_norm") is None for c in candidates):
            return candidates[:top_k]

        selected = [candidates[0]]
        pool = candidates[1:]

        while len(selected) < top_k and pool:
            # Stack vettoriale efficiente (metodo GPT)
            selected_vecs = np.stack([s["vector_norm"] for s in selected], axis=0)

            best_idx = -1
            best_mmr_score = -1e18

            for i, cand in enumerate(pool):
                rel = float(cand["h_score"])
                # Calcolo similarità vettoriale matriciale
                sims = selected_vecs @ cand["vector_norm"]
                max_sim = float(np.max(sims)) if sims.size else 0.0

                mmr = (lambda_param * rel) - ((1.0 - lambda_param) * max_sim)
                if mmr > best_mmr_score:
                    best_mmr_score = mmr
                    best_idx = i

            if best_idx == -1:
                break
            selected.append(pool.pop(best_idx))

        return selected

    def _rerank_visual(self, query_text, candidates, weight_vector=0.7, weight_bm25=0.3):
        if not candidates:
            return []

        # 0) Dedup hash veloce prima del rerank (metodo GPT)
        seen = set()
        deduped = []
        for c in candidates:
            h = hash(c["content"][:50])
            if h not in seen:
                seen.add(h)
                deduped.append(c)
        candidates = deduped

        # 1) Normalize CLIP scores (z-score + sigmoid)
        raw_scores = np.array([float(c["point"].score) for c in candidates], dtype=np.float32)
        if len(raw_scores) > 1:
            mu = float(raw_scores.mean())
            sigma = float(raw_scores.std()) + 1e-6
            z = (raw_scores - mu) / sigma
            norm_scores = (1.0 / (1.0 + np.exp(-z))).tolist()
        else:
            norm_scores = [1.0] * len(raw_scores)

        # 2) BM25 (EN)
        corpus = [c["content"] for c in candidates]
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        if tokenized_corpus:
            bm25 = BM25Okapi(tokenized_corpus)
            q_tokens = self._tokenize(query_text)
            bm25_raw = bm25.get_scores(q_tokens)
            max_bm25 = float(max(bm25_raw)) if len(bm25_raw) else 0.0
            bm25_norm = [(float(s) / max_bm25) if max_bm25 > 0 else 0.0 for s in bm25_raw]
        else:
            bm25_norm = [0.0] * len(candidates)

        processed = []
        for i, cand in enumerate(candidates):
            v_score = float(norm_scores[i])
            k_score = float(bm25_norm[i])

            # Penalità Moltiplicativa (più sicura)
            penalty = 1.0
            if self._is_sticky_content(cand["content"]):
                penalty *= 0.35 # Penalità Sticky severa
            
            # Se manca la keyword (BM25=0), penalizziamo un po' ma non uccidiamo
            if k_score == 0.0:
                penalty *= 0.85 

            hybrid = penalty * ((v_score * weight_vector) + (k_score * weight_bm25))

            # Gestione sicura del vettore (metodo GPT)
            raw_vec = getattr(cand["point"], "vector", None)
            vec_norm = self._normalize_vector(raw_vec) if raw_vec is not None else None

            processed.append({
                "point": cand["point"],
                "content": cand["content"],
                "h_score": float(hybrid),
                "vector_norm": vec_norm,
            })

        processed.sort(key=lambda x: x["h_score"], reverse=True)
        return processed

    def get_answer(self, query_text: str, top_k: int = 6): # TUNING GEMINI: top_k=6 di default
        print(f"🇮🇹 Query: {query_text}")
        english_query = self._translate_to_english(query_text)
        print(f"🇬🇧 Query Tradotta: {english_query}")

        vector_ita = self._encode(query_text)
        vector_eng = self._encode(english_query)

        initial_limit = 60
        
        # TUNING GEMINI: Quota Dinamica Aggressiva
        # Vogliamo almeno 3 slide se possibile.
        quota = max(3, top_k // 2)
        
        # Audio prende il resto, ma con un minimo di sicurezza
        audio_limit = top_k - quota + 3

        final_context = []
        seen_ids = set()
        seen_contents = set()

        # -------- VISUAL (EN) --------
        visual_candidates = self.db.search(
            query_vector=vector_eng,
            top_k=initial_limit,
            query_filter=Filter(must=[FieldCondition(key="modality", match=MatchValue(value="visual"))]),
            with_vectors=True
        )

        visual_pool = []
        for p in visual_candidates:
            content = p.payload.get("full_ocr", p.payload.get("text", ""))
            if len(content) > 15:
                visual_pool.append({"point": p, "content": content})

        scored_visuals = self._rerank_visual(english_query, visual_pool)
        
        # Passiamo la quota aggressiva a MMR
        selected_visuals = self._mmr_selection(scored_visuals, top_k=quota)

        for item in selected_visuals:
            p_id = item["point"].id
            content_hash = hash(item["content"][:50])

            if p_id not in seen_ids and content_hash not in seen_contents:
                p = item["point"].payload
                final_context.append({
                    "source": p.get("source"),
                    "timestamp": p.get("timestamp"),
                    "modality": "VISUAL",
                    "score": float(item["h_score"]),
                    "content_to_use": item["content"],
                })
                seen_ids.add(p_id)
                seen_contents.add(content_hash)

        # -------- AUDIO (ITA) --------
        audio_candidates = self.db.search(
            query_vector=vector_ita,
            top_k=audio_limit,
            query_filter=Filter(must=[FieldCondition(key="modality", match=MatchValue(value="audio"))]),
            with_vectors=False
        )

        audio_count = 0
        for anchor in audio_candidates:
            # Calcoliamo la quota audio residua dinamicamente
            if len(final_context) >= top_k:
                break
                
            if len(anchor.payload.get("text", "")) < 20: continue
            if anchor.id in seen_ids: continue

            ts = anchor.payload["timestamp"]
            src = anchor.payload["source"]

            neighbors, _ = self.db.scroll(
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="source", match=MatchValue(value=src)),
                        FieldCondition(key="modality", match=MatchValue(value="audio")),
                        FieldCondition(key="timestamp", range=Range(gt=ts - 30, lt=ts + 30)),
                    ]
                ),
                limit=15
            )

            chunk_added = False
            for n in neighbors:
                if n.id in seen_ids: continue
                p = n.payload
                if len(p.get("text", "")) > 10:
                    final_context.append({
                        "source": p.get("source"),
                        "timestamp": p.get("timestamp"),
                        "modality": "AUDIO",
                        "score": float(anchor.score),
                        "content_to_use": p.get("text", ""),
                    })
                    seen_ids.add(n.id)
                    chunk_added = True

            if chunk_added:
                audio_count += 1

        return sorted(final_context, key=lambda x: (x["source"], x["timestamp"]))