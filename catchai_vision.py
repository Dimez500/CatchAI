"""
catchai_vision.py
- Minimal wrapper around Azure AI Vision Image Analysis to get caption + tags
- Maps tags to a best-guess freshwater species for CatchAI
- Can be imported or run as a CLI:
    python catchai_vision.py --image path_or_url.jpg
Requires:
  pip install python-dotenv azure-ai-vision-imageanalysis
  and a .env with VISION_ENDPOINT, VISION_KEY
"""
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv

try:
    from azure.ai.vision.imageanalysis import ImageAnalysisClient
    from azure.ai.vision.imageanalysis.models import VisualFeatures
    from azure.core.credentials import AzureKeyCredential
except Exception:
    ImageAnalysisClient = None  # so we can raise a helpful error later

# Load species keywords from adjacent file if present
DEFAULT_SPECIES_FILE = Path(__file__).with_name("species_keywords.json")
SPECIES_KEYWORDS: Dict[str, List[str]] = {}
if DEFAULT_SPECIES_FILE.exists():
    SPECIES_KEYWORDS = json.loads(DEFAULT_SPECIES_FILE.read_text())

def best_guess_species(tags: List[Dict[str, Any]]) -> str:
    """Very simple keyword matching against tag names; returns title-cased species name or ''."""
    tag_names = [t.get("name", "").lower() for t in tags]
    joined = " ".join(tag_names)
    best = ""
    score = -1
    for species, keywords in SPECIES_KEYWORDS.items():
        local_score = sum(1 for k in keywords if k.lower() in joined)
        if local_score > score and local_score > 0:
            best = species
            score = local_score
    return best.title() if best else ""

def analyze_image(source: str, top_k: int = 7, threshold: float = 0.0) -> Dict[str, Any]:
    """Analyze a local path or URL with Azure Vision; return caption + tags + species_guess."""
    load_dotenv()
    endpoint = os.getenv("VISION_ENDPOINT")
    key = os.getenv("VISION_KEY")
    if not endpoint or not key:
        raise RuntimeError("Missing VISION_ENDPOINT or VISION_KEY in .env")

    if ImageAnalysisClient is None:
        raise RuntimeError("azure-ai-vision-imageanalysis not installed. pip install azure-ai-vision-imageanalysis")

    client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    # auto-detect URL vs file
    if source.startswith(("http://", "https://")):
        image_data = {"url": source}
    else:
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {source}")
        image_data = {"image_data": p.open("rb")}

    result = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.CAPTION, VisualFeatures.TAGS],
        gender_neutral_caption=True
    )

    # Extract caption
    caption_text = ""
    caption_conf = None
    if getattr(result, "caption", None) and result.caption.text:
        caption_text = result.caption.text
        caption_conf = result.caption.confidence

    # Extract tags
    tags_out: List[Dict[str, Any]] = []
    if getattr(result, "tags", None) and result.tags.list:
        for t in result.tags.list:
            conf = float(getattr(t, "confidence", 0.0) or 0.0)
            if conf < threshold:
                continue
            tags_out.append({"name": t.name, "confidence": conf})
    tags_out = sorted(tags_out, key=lambda x: x["confidence"], reverse=True)[:top_k]

    species = best_guess_species(tags_out)

    return {
        "source": source,
        "caption": caption_text,
        "caption_confidence": caption_conf,
        "tags": tags_out,
        "species_guess": species
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path or URL to an image")
    p.add_argument("--top-k", type=int, default=7)
    p.add_argument("--threshold", type=float, default=0.0)
    args = p.parse_args()
    info = analyze_image(args.image, top_k=args.top_k, threshold=args.threshold)
    print(json.dumps(info, indent=2))
