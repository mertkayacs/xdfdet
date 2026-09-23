import numpy as np
import pytest

from xdfdet import load_model
from xdfdet.evaluate import metrics
from xdfdet.explain import case_of, gradcam, region_scores, spread_across_regions, video_gradcam
from xdfdet.landmarks import landmarks
from xdfdet.model import build_model, video_loss


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    m = build_model(seq_len=12, backbone_weights=None)
    path = tmp_path_factory.mktemp("m") / "tiny.keras"
    m.save(path)
    return load_model(str(path))


def test_output_shape_and_range(model, face):
    from xdfdet.video import normalize
    x = np.stack([normalize(face)] * 12)[None]
    y = model.predict(x, verbose=0)
    assert y.shape == (1, 12, 1) and 0 <= y.min() and y.max() <= 1


def test_video_loss_uses_frame_mean():
    y_true = np.ones((1, 12, 1), np.float32)
    y_pred = np.full((1, 12, 1), 0.8, np.float32)
    assert np.isclose(float(video_loss(y_true, y_pred)[0]), -np.log(0.8), atol=1e-5)


def test_gradcam_and_regions(model, face):
    from xdfdet.video import normalize
    seq = np.stack([normalize(face)] * 12)
    cam = gradcam(model, seq[0])
    assert cam.shape == (224, 224) and 0 <= cam.min() and cam.max() <= 1
    assert video_gradcam(model, seq).shape == (224, 224)
    scores = region_scores(np.ones((224, 224), np.float32), landmarks(face))
    assert len(scores) == 8 and all(np.isclose(v, 100) for v in scores.values())


def test_cases_treat_fake_as_positive():
    assert case_of(0, 0.2) == "TP" and case_of(1, 0.9) == "TN"
    assert case_of(1, 0.2) == "FP" and case_of(0, 0.9) == "FN"
    assert spread_across_regions({"TP": {"mean": {"a": 10.0, "b": 30.0}}}) == {"TP": 10.0}


def test_metrics():
    m = metrics(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.3]))
    assert m["auc"] == 1 and m["f1"] == 1 and m["brier"] < 0.05


def test_lost_checkpoint_is_reported():
    with pytest.raises(ValueError, match="aug-intense"):
        load_model("aug-intense")
