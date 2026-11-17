"""Classical crypto ablation using real cryptography primitives."""

from __future__ import annotations

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def aes_encrypt(key: bytes, plaintext: bytes, nonce: bytes) -> bytes:
    return AESGCM(key).encrypt(nonce, plaintext, None)


def aes_decrypt(key: bytes, ciphertext: bytes, nonce: bytes) -> bytes:
    return AESGCM(key).decrypt(nonce, ciphertext, None)


def rsa_keypair():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private, private.public_key()


def rsa_wrap(public_key, key: bytes) -> bytes:
    return public_key.encrypt(key, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None))


def rsa_unwrap(private_key, wrapped: bytes) -> bytes:
    return private_key.decrypt(wrapped, padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None))


def ecdsa_keypair():
    private = ec.generate_private_key(ec.SECP256R1())
    return private, private.public_key()


def ecdsa_sign(private_key, payload: bytes) -> bytes:
    return private_key.sign(payload, ec.ECDSA(hashes.SHA256()))


def ecdsa_verify(public_key, signature: bytes, payload: bytes) -> bool:
    try:
        public_key.verify(signature, payload, ec.ECDSA(hashes.SHA256()))
        return True
    except Exception:
        return False


def public_key_bytes(public_key) -> bytes:
    return public_key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)

