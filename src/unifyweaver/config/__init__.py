"""UnifyWeaver configuration modules."""

from .model_registry import ModelRegistry, ModelMetadata, auto_environment

__all__ = ['ModelRegistry', 'ModelMetadata', 'auto_environment']


def load_knowledge_model(prefer_newer=None, prefer_federated=None, prefer_transformer=None):
    """
    Convenience function to load the best available knowledge model.

    Args:
        prefer_newer: Prefer more recently trained models
        prefer_federated: Prefer federated models over transformer distillations
        prefer_transformer: Prefer transformer for faster inference

    Returns:
        Tuple of (model, model_name)

    Example:
        from unifyweaver.config import load_knowledge_model
        engine, name = load_knowledge_model(prefer_newer=True)
    """
    registry = ModelRegistry()
    return registry.load_knowledge_model(
        prefer_newer=prefer_newer,
        prefer_federated=prefer_federated,
        prefer_transformer=prefer_transformer,
    )
