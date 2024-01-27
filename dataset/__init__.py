from abc import ABC, abstractmethod
from typing import Iterator

from data_definitions import Context


class Dataloader(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Context]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
