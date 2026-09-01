"""Qiskit fake-backend noise resolver."""

from __future__ import annotations


def resolve_noise_backend_name() -> str | None:
    try:
        from qiskit.providers.fake_provider import FakeNairobi  # type: ignore

        _ = FakeNairobi()
        return "FakeNairobi"
    except Exception:
        try:
            import qiskit.providers.fake_provider as fake_provider

            names = [name for name in dir(fake_provider) if name.startswith("Fake")]
            return names[0] if names else None
        except Exception:
            return None

