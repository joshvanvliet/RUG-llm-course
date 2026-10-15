from pathlib import Path

import pandas as pd
from datasets import Dataset
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from rapidfuzz import fuzz, process
from sklearn.preprocessing import MultiLabelBinarizer

from Data.util import split_labels


def normalise_sentence(sentence: str) -> str:
    return " ".join(str(sentence).casefold().split())


def find_sentence(frame: pd.DataFrame,
                   sentence: str,
                   fuzzy_threshold: int,
                   ) -> pd.DataFrame:
    normalised = frame[0].map(normalise_sentence)
    exact_matches = frame[normalised == normalise_sentence(sentence)]
    if not exact_matches.empty:
        return exact_matches

    candidates = normalised.drop_duplicates().tolist()
    match = process.extractOne(
        normalise_sentence(sentence), candidates, scorer=fuzz.ratio,
        score_cutoff=fuzzy_threshold,
    )
    if match is None:
        return frame.iloc[0:0]

    return frame[normalised == match[0]]


def create_split_dataset(english_path: Path,
                         target_path: Path,
                         alignment_path: Path,
                         seed: int = 0,
                         fuzzy_threshold: int = 90,
                        ) -> tuple[tuple[Dataset, ...], ...]:
    english_frame: pd.DataFrame = split_labels(pd.DataFrame(pd.read_csv(english_path, sep="\t", header=None)))
    target_frame: pd.DataFrame = split_labels(pd.DataFrame(pd.read_csv(target_path, sep="\t", header=None)))

    alignment_frame: pd.DataFrame = pd.DataFrame(pd.read_csv(alignment_path, sep="\t", header=None))
    alignment_frame = alignment_frame[alignment_frame[0].str.contains("en/", na=False)]

    english_rows = []
    target_rows = []
    for _, row in alignment_frame.iterrows():
        english_matches = find_sentence(
            english_frame, row[4], fuzzy_threshold
        )
        target_matches = find_sentence(
            target_frame, row[5], fuzzy_threshold
        )
        if not english_matches.empty and not target_matches.empty:
            english_rows += english_matches.values.tolist()
            target_rows += target_matches.values.tolist()

    english_frame = pd.DataFrame(english_rows)
    target_frame = pd.DataFrame(target_rows)

    return_value = []
    for data in (Dataset.from_pandas(english_frame), Dataset.from_pandas(target_frame)):
        y = MultiLabelBinarizer(classes=range(1, 9)).fit_transform(data["1"])
        
        first = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=0.3, random_state=seed
        )
        train_idx, rest_idx = next(first.split(y, y))
        
        second = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=(1/3), random_state=seed
        )
        rest_y = y[rest_idx]
        dev_relative, test_relative = next(second.split(rest_y, rest_y))
    
        train = data.select(train_idx.tolist())
        develop = data.select(rest_idx[dev_relative].tolist())
        test = data.select(rest_idx[test_relative].tolist())
        return_value.append((train, develop, test))
    
    return tuple(return_value)


if __name__ == "__main__":
    """Test Code"""
    english_path = Path("./XED/AnnotatedData/en-annotated.tsv")
    dutch_path = Path("./XED/Projections/nl-projections.tsv")
    alignment_path = Path("./XED/subtitle-retrieval/students/pairs-nl.txt")
    english_splits, dutch_splits = create_split_dataset(
        english_path, dutch_path, alignment_path
    )