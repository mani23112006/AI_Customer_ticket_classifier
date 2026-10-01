"""Tests for the synthetic sample dataset generator."""
from config.settings import CATEGORIES
from ml.make_sample_data import generate_dataset


def test_balanced_and_large_enough():
    frame = generate_dataset(per_class=200, seed=1)
    counts = frame["category"].value_counts()
    assert set(counts.index) == set(CATEGORIES)
    assert (counts == 200).all()
    assert set(frame["source"]) == {"synthetic_sample"}


def test_no_duplicates_and_varied_lengths():
    frame = generate_dataset(per_class=200, seed=1)
    assert not frame.duplicated(["subject", "message"]).any()
    lengths = frame["message"].str.len()
    assert lengths.max() > 2 * lengths.min()


def test_deterministic_for_same_seed_and_different_for_other_seed():
    a = generate_dataset(per_class=200, seed=5)
    b = generate_dataset(per_class=200, seed=5)
    c = generate_dataset(per_class=200, seed=6)
    assert a.equals(b)
    assert not a.equals(c)
