from pathlib import Path

from pandas import DataFrame, read_csv


def tsv_to_Dataframe(path: Path, language_name: str) -> DataFrame:
    data = DataFrame(read_csv(path, sep="\t", header=None))
    data.rename(columns={0: language_name, 1: "labels"}, inplace=True)
    
    return data