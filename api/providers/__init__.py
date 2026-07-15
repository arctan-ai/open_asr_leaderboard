from abc import ABC, abstractmethod
from typing import Optional


class PermanentError(Exception):
    """Error that should not be retried (e.g., URL fetch failure)."""

    pass


class APIProvider(ABC):
    @abstractmethod
    def transcribe(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        """Transcribe audio and return the text."""
        ...

    def transcribe_streaming(
        self,
        model_variant: str,
        audio_file_path: Optional[str],
        sample: dict,
        use_url: bool = False,
        language: str = "en",
        prompt: Optional[str] = None,
    ) -> str:
        """Transcribe audio through a streaming ASR endpoint and return the text."""
        raise PermanentError(
            f"Streaming ASR is not supported for {self.__class__.__name__}"
        )

    def force_streaming_for_model(self, model_variant: str) -> bool:
        """Return True when a model variant must use the streaming endpoint."""
        return False


_REGISTRY: dict[str, type[APIProvider]] = {}


def register(prefix: str):
    """Decorator to register a provider class under a model prefix."""

    def decorator(cls: type[APIProvider]):
        _REGISTRY[prefix] = cls
        return cls

    return decorator


def get_provider(model_name: str) -> tuple[APIProvider, str]:
    """Look up provider by model_name prefix, return (provider_instance, variant)."""
    for prefix, cls in _REGISTRY.items():
        if model_name.startswith(prefix + "/"):
            variant = model_name[len(prefix) + 1 :]
            return cls(), variant
    raise ValueError(
        f"No provider registered for model '{model_name}'. "
        f"Known prefixes: {list(_REGISTRY.keys())}"
    )


# Auto-import all provider modules so they register themselves
# from . import speechmatics_provider
from . import assemblyai_provider

# from . import openai_provider
# from . import elevenlabs_provider
# from . import revai_provider
# from . import aquavoice_provider
# from . import zoom_provider
from . import cartesia_provider
from . import deepgram_provider
from . import smallest_provider
from . import soniox_provider
from . import sarvam_provider
# from . import reson8_provider
from . import microsoft_azure_provider
