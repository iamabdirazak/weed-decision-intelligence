from src.acquisition.cropandweed import session_of
from src.preparation.splitting import SPLITS, split_by_session


def names():
    return [f"vwg-{s:04d}-{i:04d}" for s in range(40) for i in range(10)]


def test_field_visits_never_cross_splits():
    by_session = {}
    for n, s in split_by_session(names()).items():
        by_session.setdefault(session_of(n), set()).add(s)
    assert all(len(v) == 1 for v in by_session.values())


def test_every_split_is_close_to_its_share():
    split = split_by_session(names())
    for s, share in SPLITS.items():
        assert abs(sum(v == s for v in split.values()) / len(split) - share) <= 0.05


def test_same_seed_same_split():
    assert split_by_session(names()) == split_by_session(names())
