"""
catchai_vision.py
- Minimal wrapper around Azure AI Vision Image Analysis to get caption + tags
- Maps tags to a best-guess freshwater species for CatchAI
Usage:
  python catchai_vision.py --image <path_or_url>
Requires:
  pip install python-dotenv azure-ai-vision-imageanalysis requests
  .env with:
    VISION_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
    VISION_KEY=<your-key>
"""
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv
import requests  # fetch URL images as bytes so the SDK always gets raw bytes

try:
    from azure.ai.vision.imageanalysis import ImageAnalysisClient
    from azure.ai.vision.imageanalysis.models import VisualFeatures
    from azure.core.credentials import AzureKeyCredential
except Exception:
    ImageAnalysisClient = None  # so we can raise a helpful error later

# ---------- species keywords ----------
DEFAULT_SPECIES_FILE = Path(__file__).with_name("species_keywords.json")
SPECIES_KEYWORDS: Dict[str, List[str]] = {}
if DEFAULT_SPECIES_FILE.exists():
    SPECIES_KEYWORDS = json.loads(DEFAULT_SPECIES_FILE.read_text())

def best_guess_species(
    tags: List[Dict[str, Any]],
    caption_text: str = "",
    source: str = ""
) -> str:
    """Keyword match using tag names + caption + filename/URL; returns Title Case species or ''."""
    parts = []
    parts.append(" ".join([t.get("name", "") for t in tags]))
    if caption_text:
        parts.append(caption_text)
    if source:
        # include filename and URL path for hints like "Largemouth_Bass_fish.jpg"
        try:
            from urllib.parse import urlparse
            path = urlparse(source).path if source.startswith(("http://", "https://")) else str(Path(source).name)
        except Exception:
            path = str(source)
        parts.append(path.replace("_", " ").replace("-", " "))

    haystack = " ".join(parts).lower()

    best, score = "", -1
    for species, keywords in SPECIES_KEYWORDS.items():
        s = sum(1 for k in keywords if k.lower() in haystack)
        if s > score and s > 0:
            best, score = species, s

    if best:
        return best.title()

    # Fallback: catch general bass terms
    if "bass" in haystack:
        if "large mouth" in haystack or "largemouth" in haystack:
            return "Largemouth Bass"
        if "smallmouth" in haystack or "small mouth" in haystack:
            return "Smallmouth Bass"
        if "striped" in haystack or "striper" in haystack:
            return "Striped Bass"
        if "white bass" in haystack or "sand bass" in haystack:
            return "White Bass"
        if "spotted bass" in haystack or "kentucky bass" in haystack or "alabama bass" in haystack:
            return "Spotted Bass"
        return "Bass"

    return ""

def analyze_image(source: str, top_k: int = 7, threshold: float = 0.0) -> Dict[str, Any]:
    """Analyze a local path or URL with Azure Vision; return caption + tags + species_guess."""
    # Load .env next to this file
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

    endpoint = os.getenv("VISION_ENDPOINT")
    key = os.getenv("VISION_KEY")
    if not endpoint or not key:
        raise RuntimeError("Missing VISION_ENDPOINT or VISION_KEY in .env")

    if ImageAnalysisClient is None:
        raise RuntimeError("azure-ai-vision-imageanalysis not installed. pip install azure-ai-vision-imageanalysis")

    client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    # ---- Always send raw bytes to Azure (works across SDK versions) ----
    if source.startswith(("http://", "https://")):
        resp = requests.get(source, timeout=20, headers={"User-Agent": "CatchAI/1.0"})
        resp.raise_for_status()
        image_bytes = resp.content
        if not image_bytes:
            raise RuntimeError("Downloaded image has no content (0 bytes).")
    else:
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {source}")
        image_bytes = p.read_bytes()

    # Call Azure with bytes
    result = client.analyze(
        image_data=image_bytes,
        visual_features=[VisualFeatures.CAPTION, VisualFeatures.TAGS],
        gender_neutral_caption=True,
    )

    # ---- Extract caption ----
    caption_text = ""
    caption_conf = None
    if getattr(result, "caption", None) and result.caption.text:
        caption_text = result.caption.text
        caption_conf = result.caption.confidence

    # ---- Extract tags ----
    tags_out: List[Dict[str, Any]] = []
    if getattr(result, "tags", None) and result.tags.list:
        for t in result.tags.list:
            conf = float(getattr(t, "confidence", 0.0) or 0.0)
            if conf >= threshold:
                tags_out.append({"name": t.name, "confidence": conf})
    tags_out = sorted(tags_out, key=lambda x: x["confidence"], reverse=True)[:top_k]

    # ---- Species guess + fallback ----
    species = best_guess_species(tags_out, caption_text, source)
    if not species:
        text = (caption_text + " " + " ".join(t["name"] for t in tags_out)).lower()
        if "bass" in text:
            species = "Bass"

    return {
        "source": source,
        "caption": caption_text,
        "caption_confidence": caption_conf,
        "tags": tags_out,
        "species_guess": species,
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path or URL to an image")
    p.add_argument("--top-k", type=int, default=7)
    p.add_argument("--threshold", type=float, default=0.0)
    args = p.parse_args()
    info = analyze_image(args.image, top_k=args.top_k, threshold=args.threshold)
    print(json.dumps(info, indent=2))
