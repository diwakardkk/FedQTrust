"""Measured authenticated transport for the experiment simulator.

KEM establishes an AES-GCM key; it is not itself message encryption or
privacy-preserving secure aggregation. Server sees decrypted individual updates.
"""
import os
import time
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from .classical_crypto import rsa_keypair, rsa_wrap, rsa_unwrap, ecdsa_keypair, ecdsa_sign, ecdsa_verify, public_key_bytes

class Transport:

    def __init__(self, mode):
        self.mode = mode
        self.client_signers = {}
        if mode in ('pqc', 'pqc_sign'):
            # 'pqc': ML-KEM-768 + ML-DSA-65 (FedQTrust, SecEdge-MC).
            # 'pqc_sign': ML-DSA-65 signatures only, no KEM (PQS-BFL, manuscript Table 4).
            import oqs
            if mode == 'pqc':
                self.kem = oqs.KeyEncapsulation('ML-KEM-768')
                self.kem_public = self.kem.generate_keypair()
            self.signer = oqs.Signature('ML-DSA-65')
            self.sig_public = self.signer.generate_keypair()
        elif mode == 'classical':
            self.rsa_private, self.rsa_public = rsa_keypair()
            self.sig_private, self.sig_public = ecdsa_keypair()
        elif mode != 'none':
            raise ValueError(mode)

    def roundtrip(self, payload: bytes, context: bytes):
        start = time.perf_counter()
        if self.mode in ('none', 'pqc_sign'):
            return (payload, dict(transport_s=0.0, update_bytes=len(payload), wire_bytes=len(payload)))
        nonce = os.urandom(12)
        if self.mode == 'pqc':
            import oqs
            with oqs.KeyEncapsulation('ML-KEM-768') as sender:
                wrapped, key = sender.encap_secret(self.kem_public)
            recv = self.kem.decap_secret(wrapped)
        else:
            key = os.urandom(32)
            wrapped = rsa_wrap(self.rsa_public, key)
            recv = rsa_unwrap(self.rsa_private, wrapped)

        def derive(secret):
            return HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b'FedQTrust update v1'
            ).derive(secret)
        cipher = AESGCM(derive(key)).encrypt(nonce, payload, context)
        decoded = AESGCM(derive(recv)).decrypt(nonce, cipher, context)
        if decoded != payload:
            raise RuntimeError('Update transport mismatch')
        return (
            decoded,
            dict(
                transport_s=time.perf_counter() - start,
                update_bytes=len(payload),
                wire_bytes=len(cipher) + len(wrapped) + len(nonce)
            )
        )

    def authenticate(self, payload: bytes, signer_id='server'):
        import hashlib
        payload = hashlib.sha256(payload).digest()
        if self.mode == 'none':
            return dict(sign_verify_s=0.0, signature_bytes=0)
        # One-time client key generation is enrollment, not per-round signing cost.
        if signer_id != 'server' and signer_id not in self.client_signers:
            if self.mode in ('pqc', 'pqc_sign'):
                import oqs
                signer = oqs.Signature('ML-DSA-65')
                self.client_signers[signer_id] = (signer, signer.generate_keypair())
            else:
                self.client_signers[signer_id] = ecdsa_keypair()
        start = time.perf_counter()
        if self.mode in ('pqc', 'pqc_sign'):
            import oqs
            if signer_id == 'server':
                signer, public = (self.signer, self.sig_public)
            else:
                signer, public = self.client_signers[signer_id]
            sig = signer.sign(payload)
            with oqs.Signature('ML-DSA-65') as verifier:
                valid = verifier.verify(payload, sig, public)
        else:
            if signer_id == 'server':
                private, public_key = (self.sig_private, self.sig_public)
            else:
                private, public_key = self.client_signers[signer_id]
            sig = ecdsa_sign(private, payload)
            valid = ecdsa_verify(public_key, sig, payload)
            public = public_key_bytes(public_key)
        if not valid:
            raise RuntimeError('Global model signature rejected')
        result = dict(sign_verify_s=time.perf_counter() - start, signature_bytes=len(sig))
        result['proof'] = dict(
            signer=str(signer_id),
            digest=payload.hex(),
            signature=sig.hex(),
            public_key=public.hex(),
            scheme='ML-DSA-65' if self.mode in ('pqc', 'pqc_sign') else 'ECDSA-P256'
        )
        return result

    @staticmethod
    def verify(payload, proof):
        import hashlib
        digest = hashlib.sha256(payload).digest()
        if digest.hex() != proof['digest']:
            raise ValueError('Signed payload digest mismatch')
        signature = bytes.fromhex(proof['signature'])
        public = bytes.fromhex(proof['public_key'])
        if proof['scheme'] == 'ML-DSA-65':
            import oqs
            with oqs.Signature('ML-DSA-65') as verifier:
                valid = verifier.verify(digest, signature, public)
        elif proof['scheme'] == 'ECDSA-P256':
            from cryptography.hazmat.primitives.serialization import load_der_public_key
            valid = ecdsa_verify(load_der_public_key(public), signature, digest)
        else:
            raise ValueError('Unknown signature scheme')
        if not valid:
            raise ValueError('Signature verification failed')
        return True
