"""Owner-approved browser session import with an ephemeral cloud-only private key."""
import base64
import json
import time

import requests
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Hash import SHA256

from src.api.email_relay import EmailRelay
from src.api.session_cache import save_session
from src.api.webvpn import WebVPNSession, get_vpn_url
from src.runtime import config


class SessionImportRelay(EmailRelay):
    workflow_path = '.github/workflows/session_import.yml'
    artifact_prefix = 'session-response-'
    message_limit = 60000
    archive_limit = 65000

    def decode(self, ciphertext):
        try:
            if not isinstance(ciphertext, str) or len(ciphertext) > self.message_limit:
                raise ValueError()
            envelope = json.loads(ciphertext)
            parts = {k: base64.b64decode(envelope[k], validate=True)
                     for k in ('key', 'nonce', 'tag', 'ciphertext')}
            if len(parts['nonce']) != 12 or len(parts['tag']) != 16:
                raise ValueError()
            key = PKCS1_OAEP.new(self.key, hashAlgo=SHA256,
                                label=self.target.encode()).decrypt(parts['key'])
            if len(key) != 32:
                raise ValueError()
            aes = AES.new(key, AES.MODE_GCM, nonce=parts['nonce'])
            aes.update(self.target.encode())
            value = json.loads(aes.decrypt_and_verify(parts['ciphertext'], parts['tag']))
            if (value.get('target') != self.target or value.get('action') != 'session'
                    or not 0 <= time.time() - value['created_at'] <= 600):
                raise ValueError()
            value['cookies'] = normalize_cookies(value['cookies'])
            return value
        except (ValueError, TypeError, KeyError, UnicodeError, AttributeError):
            raise ValueError('Invalid encrypted session import') from None


def normalize_cookies(cookies):
    # Limit browser import to the WebVPN origin, including proxied iCourse cookies.
    if not isinstance(cookies, list) or not 1 <= len(cookies) <= 100:
        raise ValueError('Invalid cookie collection')
    result = []
    for item in cookies:
        if (not isinstance(item, dict)
                or item.get('domain') not in ('webvpn.fudan.edu.cn', '.webvpn.fudan.edu.cn')
                or not isinstance(item.get('name'), str) or not item['name']
                or not isinstance(item.get('value'), str)
                or len(item['name']) + len(item['value']) > 12000
                or not isinstance(item.get('path'), str) or not item['path'].startswith('/')
                or type(item.get('secure')) is not bool):
            raise ValueError('Invalid cookie scope')
        expires = item.get('expires')
        if expires is not None:
            if type(expires) not in (int, float):
                raise ValueError('Invalid cookie expiration')
            expires = None if expires < 0 else int(expires)
            if expires is not None and expires <= time.time():
                continue
        result.append(dict(name=item['name'], value=item['value'], domain=item['domain'],
                           path=item['path'], secure=item['secure'], expires=expires))
    if not result:
        raise ValueError('No unexpired cookies')
    return result


def verify_and_save(cookies):
    vpn = WebVPNSession()
    try:
        for item in normalize_cookies(cookies):
            vpn.session.cookies.set_cookie(requests.cookies.create_cookie(**item))
        # Do not follow login redirects: an import must already be authenticated.
        for attempt in range(3):
            try:
                reply = vpn.session.get(get_vpn_url(config.ICOURSE_BASE + '/userapi/v1/infosimple'),
                                        timeout=(15, 45), allow_redirects=False)
                status = reply.status_code if type(reply.status_code) is int else 0
                print('[SessionImport] Cloud API HTTP status: ' + str(status), flush=True)
                vpn._validate_icourse_response(reply)
                break
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 2:
                    raise
                time.sleep(5)
        vpn.logged_in = True
        if not save_session(vpn):
            raise ValueError('Encrypted cache path is required')
        print('[SessionImport] Cloud iCourse API verified; encrypted session saved.', flush=True)
    finally:
        vpn.session.close()
