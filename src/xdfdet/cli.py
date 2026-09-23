"""Command line: xdfdet {predict, crop, train, evaluate, analyze}."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from .config import CONFIGS  # noqa: E402

DEFAULT_MODEL = "aug-cutout-black"


def _splits(args):
    from .data import pair_videos, split_pairs
    return split_pairs(pair_videos(args.data), seed=args.seed)


def cmd_predict(args):
    import numpy as np

    from .explain import overlay, region_scores, video_gradcam
    from .landmarks import landmarks
    from .model import load_model, video_score
    from .video import crop_faces, frame_indices, normalize

    crops = crop_faces(args.video)
    frames = [crops[i] for i in frame_indices(len(crops))]
    seq = np.stack([normalize(f) for f in frames])
    model = load_model(args.model)
    score = video_score(model, seq)
    out = {"video": str(args.video), "model": args.model, "real_probability": round(score, 4),
           "verdict": "real" if score >= 0.5 else "fake"}
    if args.gradcam:
        import cv2
        cam = video_gradcam(model, seq)
        mid = frames[len(frames) // 2]
        cv2.imwrite(str(args.gradcam), cv2.cvtColor(overlay(mid, cam), cv2.COLOR_RGB2BGR))
        points = landmarks(mid)
        if points is not None:
            out["regions"] = {k: round(v, 1) for k, v in region_scores(cam, points).items()}
        out["gradcam"] = str(args.gradcam)
    print(json.dumps(out, indent=2))


def cmd_crop(args):
    from .video import crop_faces, write_video
    src, dst = Path(args.src), Path(args.dst)
    videos = sorted(src.rglob("*.mp4"))
    for i, path in enumerate(videos, 1):
        target = dst / path.relative_to(src)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            write_video(crop_faces(path), target)
        except ValueError as err:
            print(f"skip: {err}", file=sys.stderr)
        print(f"[{i}/{len(videos)}] {path.relative_to(src)}", file=sys.stderr)


def cmd_train(args):
    from .train import train
    train_pairs, val_pairs, _ = _splits(args)
    train(args.config, train_pairs, val_pairs, args.out)


def cmd_evaluate(args):
    from .evaluate import metrics, score_pairs
    from .model import load_model
    *_, test = _splits(args)
    labels, scores = score_pairs(load_model(args.model, mixed_precision=args.mixed_precision), test)
    print(json.dumps({k: round(float(v), 4) for k, v in metrics(labels, scores).items()}, indent=2))


def cmd_analyze(args):
    from .explain import analyze, spread_across_regions
    from .model import load_model
    *_, test = _splits(args)
    result = analyze(load_model(args.model), test, max_per_case=args.max_per_case)
    result["spread_across_regions"] = spread_across_regions(result)
    text = json.dumps(result, indent=2)
    Path(args.out).write_text(text) if args.out else print(text)


def main(argv=None):
    p = argparse.ArgumentParser(prog="xdfdet", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    models = f"a released name ({', '.join(CONFIGS)}) or a .keras file"

    s = sub.add_parser("predict", help="score one video and optionally save a Grad-CAM overlay")
    s.add_argument("video")
    s.add_argument("--model", default=DEFAULT_MODEL, help=models)
    s.add_argument("--gradcam", metavar="PNG", help="where to write the Grad-CAM overlay")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("crop", help="MTCNN face crops (32 frames, 224x224) for a video tree")
    s.add_argument("src")
    s.add_argument("dst")
    s.set_defaults(func=cmd_crop)

    for name, func, text in (("train", cmd_train, "train one configuration"),
                             ("evaluate", cmd_evaluate, "AUC, F1, Brier, LogLoss on the test split"),
                             ("analyze", cmd_analyze, "Grad-CAM region analysis on the test split")):
        s = sub.add_parser(name, help=text)
        s.add_argument("--data", required=True, help="folder produced by `xdfdet crop`")
        s.add_argument("--seed", type=int, default=42, help="split seed")
        s.set_defaults(func=func)
        if name == "train":
            s.add_argument("--config", required=True, choices=list(CONFIGS))
            s.add_argument("--out", required=True, help="output .keras path")
        else:
            s.add_argument("--model", default=DEFAULT_MODEL, help=models)
        if name == "evaluate":
            s.add_argument("--mixed-precision", action="store_true",
                           help="run in mixed float16 like the original runs (GPU)")
        if name == "analyze":
            s.add_argument("--max-per-case", type=int, default=50)
            s.add_argument("--out", help="write JSON here instead of stdout")

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
