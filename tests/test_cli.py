import json

from xdfdet.cli import main
from xdfdet.model import build_model


def test_predict_end_to_end(clip_tree, tmp_path, capsys):
    model = tmp_path / "m.keras"
    build_model(backbone_weights=None).save(model)
    main(["predict", str(clip_tree / "000.mp4"), "--model", str(model), "--gradcam", str(tmp_path / "cam.png")])
    out = json.loads(capsys.readouterr().out)
    assert 0 <= out["real_probability"] <= 1 and out["verdict"] in ("real", "fake")
    assert (tmp_path / "cam.png").exists() and len(out["regions"]) == 8
