from xdfdet.data import pair_videos, split_pairs


def test_pairs_cycle_manipulations_with_fallback(clip_tree):
    pairs = pair_videos(clip_tree, manipulations=("FaceSwap", "Face2Face"))
    assert len(pairs) == 8
    for i, (real, fake) in enumerate(pairs):
        assert fake.split("/")[-1].startswith(real.split("/")[-1][:3] + "_")
    # video 001 prefers FaceSwap but only has Face2Face: the fallback keeps it
    assert "Face2Face" in pairs[1][1]


def test_split_is_seeded_and_disjoint():
    pairs = [(f"r{i}", f"f{i}") for i in range(100)]
    a = split_pairs(pairs, seed=42)
    assert a == split_pairs(pairs, seed=42)
    assert a != split_pairs(pairs, seed=1)
    train, val, test = a
    assert (len(train), len(val), len(test)) == (70, 15, 15)
    assert not set(train) & set(val) | set(train) & set(test) | set(val) & set(test)
