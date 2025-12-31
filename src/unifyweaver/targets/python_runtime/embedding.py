import abc
from typing import List

class IEmbeddingProvider(abc.ABC):
    @abc.abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        pass
