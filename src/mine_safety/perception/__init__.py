from .anomaly import anomaly_nodes_from_probabilities, predictive_entropy

__all__ = [
    "ContrastiveMinkUNet",
    "SemanticMinkUNet",
    "anomaly_nodes_from_probabilities",
    "predictive_entropy",
]


def __getattr__(name: str):
    """Keep CPU-only graph/test installs independent of the optional PyTorch stack."""
    if name in {"ContrastiveMinkUNet", "SemanticMinkUNet"}:
        from .network import ContrastiveMinkUNet, SemanticMinkUNet

        return {"ContrastiveMinkUNet": ContrastiveMinkUNet, "SemanticMinkUNet": SemanticMinkUNet}[name]
    raise AttributeError(name)
