import abc

class IEmbeddingProvider(abc.ABC):
    @abc.abstractmethod
    def get_embedding(self, text: str) -> list[float]:
        pass
