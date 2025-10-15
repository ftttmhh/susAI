from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional

# Default model is lightweight but good for semantic similarity. You can switch to a
# smaller / faster model like sentence-transformers/all-MiniLM-L6-v2 for production.
MODEL_NAME = "distilbert-base-uncased"


class BERTTaskClassifier:
    def __init__(self, model_name: str = MODEL_NAME, device: str = None, min_score: float = 0.45):
        """Create a classifier that maps user text to one of the provided categories.

        Improvements included:
        - lowercase and simple token checks for short prompts
        - thresholding to avoid mapping to unrelated categories
        - caching of category embeddings
        - optional model override
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception:
            # If HF model download fails or is too heavy, raise to let caller decide
            raise

        self.cached_category_embeddings: Optional[np.ndarray] = None
        self.cached_categories: List[str] = []
        self.min_score = float(min_score)

        # simple keyword map to quickly handle short prompts and single-word intents
        self.keyword_map = {
            "draw": "Image Generation",
            "picture": "Image Generation",
            "image": "Image Generation",
            "photo": "Image Generation",
            "animate": "Video Generation",
            "animation": "Video Generation",
            "summarize": "Summarization",
            "summarise": "Summarization",
            "translate": "Translation",
            "classify": "Text Classification",
            "sentiment": "Text Classification",
            "answer": "Question Answering",
            "qa": "Question Answering",
            "speech": "Automatic Speech Recognition",
            "asr": "Automatic Speech Recognition",
            "caption": "Image Captioning",
            "embed": "Sentence Similarity",
            "similarity": "Sentence Similarity",
        }

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            masked_hidden = last_hidden * mask
            summed = masked_hidden.sum(dim=1)
            counts = mask.sum(dim=1)
            counts = torch.clamp(counts, min=1e-9)
            mean_pooled = summed / counts
            embeddings = mean_pooled.cpu().numpy()
        return embeddings

    def fit_categories(self, category_texts: List[str]):
        # store canonical categories and precompute embeddings
        cats = [str(c).strip() for c in category_texts if c is not None]
        self.cached_categories = cats
        if len(cats) > 0:
            self.cached_category_embeddings = self.embed_texts(cats)
        else:
            self.cached_category_embeddings = None

    def _keyword_match(self, text: str) -> Optional[str]:
        t = text.strip().lower()
        # exact word match or token prefix
        for k, v in self.keyword_map.items():
            if k == t or k in t.split() or t.startswith(k):
                return v
        return None

    def predict(self, user_text: str, top_k: int = 1):
        """Return top_k category predictions with similarity scores.

        For short inputs or single words, a keyword-based match is attempted first.
        A similarity threshold (self.min_score) is applied to avoid poor mappings.
        """
        if not user_text or not str(user_text).strip():
            return []

        # quick keyword fallback for very short inputs
        if len(user_text.strip().split()) <= 3:
            km = self._keyword_match(user_text)
            if km:
                return [{"category": km, "score": 1.0}]

        if self.cached_category_embeddings is None:
            raise RuntimeError("fit_categories must be called before predict()")

        emb = self.embed_texts([user_text])
        sims = cosine_similarity(emb, self.cached_category_embeddings)[0]
        # normalize similarity to 0..1 range (cosine already -1..1)
        sims = (sims + 1.0) / 2.0
        idxs = np.argsort(-sims)
        results = []
        for i in idxs[: int(top_k)]:
            score = float(sims[i])
            if score < self.min_score:
                # skip if below threshold
                continue
            results.append({"category": self.cached_categories[i], "score": score})

        # if thresholding removed all candidates, fall back to top raw match
        if not results and len(idxs) > 0:
            i = idxs[0]
            results = [{"category": self.cached_categories[i], "score": float(sims[i])}]

        return results

    def predict_many(self, texts: List[str], top_k: int = 1):
        return [self.predict(t, top_k=top_k) for t in texts]
