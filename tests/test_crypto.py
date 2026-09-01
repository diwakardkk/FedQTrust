from os import urandom

from fedqtrust.security.classical_crypto import aes_decrypt, aes_encrypt, ecdsa_keypair, ecdsa_sign, ecdsa_verify, rsa_keypair, rsa_unwrap, rsa_wrap
from fedqtrust.security.pqc import enabled_mechanisms


def test_classical_crypto_round_trip_and_tamper_rejection():
    key = urandom(32)
    nonce = urandom(12)
    payload = b"fedqtrust"
    ciphertext = aes_encrypt(key, payload, nonce)
    assert aes_decrypt(key, ciphertext, nonce) == payload
    rsa_private, rsa_public = rsa_keypair()
    wrapped = rsa_wrap(rsa_public, key)
    assert rsa_unwrap(rsa_private, wrapped) == key
    sig_private, sig_public = ecdsa_keypair()
    signature = ecdsa_sign(sig_private, payload)
    assert ecdsa_verify(sig_public, signature, payload)
    assert not ecdsa_verify(sig_public, signature, payload + b"!")


def test_pqc_enumeration_shape():
    mechs = enabled_mechanisms()
    assert "kem" in mechs and "sig" in mechs

