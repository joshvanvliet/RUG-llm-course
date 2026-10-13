from pathlib import Path

import pandas as pd
from datasets import Dataset

from Data.utils import tsv_to_Dataframe


def create_split_dataset(first_lang_path: Path,
                         second_lang_path: Path,
                         seed: int = 0) -> tuple[Dataset, ...]:
    bigger_frame = tsv_to_Dataframe(first_lang_path, "English")
    smaller_frame = tsv_to_Dataframe(second_lang_path, "Dutch")

    if len(smaller_frame) > len(bigger_frame):
        bigger_frame, smaller_frame = smaller_frame, bigger_frame
    
    return ()


if __name__ == "__main__":
    """Test Code"""
    from Data.postprocesses import *
    from Data.preprocesses import *
    
    eng_test_data_path = Path("./XED/AnnotatedData/en-annotated.tsv")
    nl_test_data_path = Path("./XED/Projections/nl-projections.tsv")
    create_split_dataset(eng_test_data_path, nl_test_data_path)
    