"""Render the saved RAGAS report as a terminal-style evidence image."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "data" / "ragas_report.json"
OUTPUT = ROOT / "evidence" / "03_ragas_scores.png"


def font(size: int, bold: bool = False):
    name = "consolab.ttf" if bold else "consola.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def main():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    v1 = report["prompt_v1_scores"]
    v2 = report["prompt_v2_scores"]
    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    ]

    image = Image.new("RGB", (1280, 620), "#0c0f14")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1280, 48), fill="#20252d")
    draw.ellipse((18, 16, 34, 32), fill="#ff5f57")
    draw.ellipse((44, 16, 60, 32), fill="#febc2e")
    draw.ellipse((70, 16, 86, 32), fill="#28c840")
    draw.text((105, 12), "RAGAS Evaluation - Ollama local run", font=font(20), fill="#d8dee9")

    draw.text((48, 82), "Bước 3: RAGAS Evaluation", font=font(30, True), fill="#f8fafc")
    draw.text((48, 130), "50 QA pairs x 2 prompt versions", font=font(19), fill="#94a3b8")
    draw.line((48, 180, 1230, 180), fill="#475569", width=2)

    draw.text((68, 207), "Metric", font=font(21, True), fill="#e2e8f0")
    draw.text((650, 207), "V1", font=font(21, True), fill="#7dd3fc")
    draw.text((850, 207), "V2", font=font(21, True), fill="#fda4af")
    draw.text((1040, 207), "Winner", font=font(21, True), fill="#e2e8f0")

    y = 265
    for metric in metrics:
        winner = "V1" if v1[metric] > v2[metric] else "Tie"
        draw.text((68, y), metric, font=font(20), fill="#e2e8f0")
        draw.text((650, y), f"{v1[metric]:.4f}", font=font(20), fill="#bae6fd")
        draw.text((850, y), f"{v2[metric]:.4f}", font=font(20), fill="#fecdd3")
        draw.text((1040, y), winner, font=font(20), fill="#86efac")
        y += 62

    best = max(v1["faithfulness"], v2["faithfulness"])
    draw.line((48, 510, 1230, 510), fill="#475569", width=2)
    draw.text(
        (48, 538),
        f"PASS  Best faithfulness = {best:.4f} >= 0.8",
        font=font(23, True),
        fill="#4ade80",
    )
    image.save(OUTPUT)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
