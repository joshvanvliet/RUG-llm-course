# guys... please start using tests

import random

import pytest

from Data.dataset_maker import create_split_dataset, normalise_sentence
from Data.paths import XED_DIR


def get_sentences(dataset):
    return {normalise_sentence(sentence) for sentence in dataset["sentence"]}


def check_split(english_splits, target_splits):
    # parallel lines SHOULD be in the same partition
    for english, target in zip(english_splits, target_splits):
        assert english["id"] == target["id"]

    # duped sentences should not cross partitions
    for splits in (english_splits, target_splits):
        train, develop, test = [get_sentences(dataset) for dataset in splits]
        assert not train & develop
        assert not train & test
        assert not develop & test


def test_fake_data(tmp_path):
    rng = random.Random(0)
    english_lines, dutch_lines, pair_lines = [], [], []
    for i in range(300):
        labels = ",".join(str(label) for label in sorted(rng.sample(range(1, 9), 2)))
        # every 10th english and every 11th dutch sentence is a test duplicate
        english = f"English sentence {i}" if i % 10 else "Hello there"
        dutch = f"Nederlandse zin {i}" if i % 11 else "Hallo daar"
        english_lines.append(f"{english}\t{labels}")
        dutch_lines.append(f"{dutch}\t{labels}")
        pair_lines.append(f"en/{i}.xml.gz\tnl/{i}.xml.gz\t{i}\t{i}\t{english}\t{dutch}")

    (tmp_path / "en.tsv").write_text("\n".join(english_lines) + "\n", encoding="utf-8")
    (tmp_path / "nl.tsv").write_text("\n".join(dutch_lines) + "\n", encoding="utf-8")
    (tmp_path / "pairs.txt").write_text("\n".join(pair_lines) + "\n", encoding="utf-8")

    english_splits, dutch_splits = create_split_dataset(
        tmp_path / "en.tsv", tmp_path / "nl.tsv", tmp_path / "pairs.txt", tmp_path / "split.tsv"
    )

    check_split(english_splits, dutch_splits)


@pytest.mark.skipif(not (XED_DIR / "AnnotatedData").exists(), reason="XED submodule not checked out")
def test_real_dutch_data(tmp_path):
    english_splits, dutch_splits = create_split_dataset(
        XED_DIR / "AnnotatedData" / "en-annotated.tsv",
        XED_DIR / "Projections" / "nl-projections.tsv",
        XED_DIR / "subtitle-retrieval" / "students" / "pairs-nl.txt",
        tmp_path / "nl.tsv",
    )

    check_split(english_splits, dutch_splits)
