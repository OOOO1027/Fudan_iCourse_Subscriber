"""Explicit interactive second-factor completion using the official IDP flow."""
import base64
import hashlib
import time

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import requests

from src.api.webvpn import AuthenticationActionRequired, require_login_token


def seal(data, student_id, password, purpose):
    salt = get_random_bytes(16)
    key = hashlib.pbkdf2_hmac('sha256', (student_id + ':' + password).encode(), salt, 200000)
    cipher = AES.new(key, AES.MODE_GCM, nonce=get_random_bytes(12))
    cipher.update(purpose.encode())
    encrypted, tag = cipher.encrypt_and_digest(data)
    return b'ICSA1' + salt + cipher.nonce + tag + encrypted


def unseal(data, student_id, password, purpose):
    if data[:5] != b'ICSA1' or len(data) < 49:
        raise ValueError('Invalid encrypted authentication artifact')
    key = hashlib.pbkdf2_hmac('sha256', (student_id + ':' + password).encode(), data[5:21], 200000)
    cipher = AES.new(key, AES.MODE_GCM, nonce=data[21:33])
    cipher.update(purpose.encode())
    return cipher.decrypt_and_verify(data[49:], data[33:49])


class QRVerification:
    def __init__(self, publish):
        self.publish = publish

    def __call__(self, session, url, headers, previous, challenge):
        if 'qr' not in (challenge.get('moduleCodes') or []):
            raise AuthenticationActionRequired('QR verification is not offered by the identity provider')
        qr = session.post(url.rsplit('/', 1)[0] + '/getAuthQr',
                          json={'entityId': previous['entityId']}, headers=headers, timeout=60)
        qr.raise_for_status()
        reply = qr.json()
        if str(reply.get('code')) != '200' or not isinstance(reply.get('data'), dict):
            raise AuthenticationActionRequired('The identity provider could not generate a QR challenge')
        qr_data = reply['data']
        image = qr_data.get('qrMsg', '')
        token = qr_data.get('qrToken')
        if not isinstance(image, str) or not image.startswith('data:image/') or not token:
            raise AuthenticationActionRequired('Unexpected official QR response format')
        prefix, encoded = image.split(',', 1)
        if ';base64' not in prefix or len(encoded) > 2_000_000:
            raise AuthenticationActionRequired('Invalid QR image encoding')
        image_bytes = base64.b64decode(encoded, validate=True)
        payload = {
            'authModuleCode': 'qr', 'authChainCode': challenge['authChainCode'],
            'entityId': previous['entityId'], 'requestNumber': challenge['requestNumber'],
            'requestType': previous['requestType'], 'lck': previous['lck'],
            'authPara': {'loginName': challenge['userName'], 'qrToken': token},
        }
        self.publish(image_bytes)
        print('[QR] Waiting for official scan confirmation; no verification data logged.', flush=True)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            try:
                response = session.post(url, json=payload, headers=headers, timeout=(15, 180))
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.ReadTimeout:
                continue
            require_login_token(result, 'Official QR verification')
            print('[QR] Official second-factor verification completed.', flush=True)
            return result
        raise AuthenticationActionRequired('QR verification timed out; a fresh official challenge is required')
