"""TrustLens: detect AI-written / suspicious product reviews and summarise what real customers say."""
from .aspects import ASPECTS, find_aspects, split_sentences, aspect_inputs  # noqa: F401
from .scoring import trust_adjusted_rating, summarise_aspects  # noqa: F401

__version__ = "0.1.0"
