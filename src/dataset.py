from collections.abc import Callable

from pandas import DataFrame
from torch.utils.data import Dataset


class VacanciesDataset(Dataset):
    def __init__(
        self,
        dataframe: DataFrame,
        processing_func: Callable | None = None,
    ) -> None:
        self._data = dataframe
        self._processing_func = processing_func

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> dict:
        row = self._data.iloc[index]
        text = row["description"]
        label = row["staff_type"]

        if self._processing_func is not None:
            text = self._processing_func(text)

        return (text, label)
