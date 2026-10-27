"""Select numerical implementations using the input tensors' device."""

from .numpy_backend import NumPyBackend


_BACKENDS = {"cpu": NumPyBackend()}

def get_backend(device):
    """Return the backend for a supported device; never silently fall back."""
    if device not in _BACKENDS:
        raise NotImplementedError(f"No backend is available for device {device!r}.")
    return _BACKENDS[device]
