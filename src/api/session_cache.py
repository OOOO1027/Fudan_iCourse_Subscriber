"""Authenticated encrypted session cache, opt-in via a runner-temporary path."""
import json
import os
from pathlib import Path
import time

import requests
from src.api.qr_verification import seal, unseal
from src.api.webvpn import WebVPNSession, get_vpn_url
from src.runtime import config

DOMAINS = {'webvpn.fudan.edu.cn', 'id.fudan.edu.cn', 'icourse.fudan.edu.cn', 'fudan.edu.cn'}


def cache_path():
    value = os.environ.get('ICOURSE_SESSION_FILE')
    if not value:
        return None
    path = Path(value).resolve()
    runner = os.environ.get('RUNNER_TEMP')
    if not runner or Path(runner).resolve() not in path.parents:
        raise ValueError('Session cache must be under RUNNER_TEMP, outside the checkout')
    return path


def save_session(vpn):
    path = cache_path()
    if path is None:
        return False
    if not config.STUDENT_ID or not config.PASSWORD:
        raise ValueError('Missing cache encryption credentials')
    cookies = []
    for c in vpn.session.cookies:
        if c.domain.lstrip('.') in DOMAINS:
            cookies.append({k: getattr(c, k) for k in
                            ('name', 'value', 'domain', 'path', 'secure', 'expires')})
    payload = json.dumps({'version': 1, 'saved_at': time.time(), 'cookies': cookies}).encode()
    encrypted = seal(payload, config.STUDENT_ID, config.PASSWORD, 'session-cache')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_bytes(encrypted)
    temporary.chmod(0o600)
    temporary.replace(path)
    path.with_suffix('.verified').write_text('verified\n')
    print('[SessionCache] Saved encrypted session; no plaintext session stored.', flush=True)
    return True


def load_verified_session():
    path = cache_path()
    if path is None or not path.is_file():
        return None
    vpn = None
    try:
        if path.stat().st_size > 1_000_000:
            raise ValueError('Cache too large')
        payload = json.loads(unseal(path.read_bytes(), config.STUDENT_ID,
                                    config.PASSWORD, 'session-cache'))
        if (payload['version'] != 1 or not 0 <= time.time() - payload['saved_at'] <= 86400
                or not isinstance(payload['cookies'], list)):
            raise ValueError('Cache invalid or older than one day')
        vpn = WebVPNSession()
        for item in payload['cookies']:
            if (not isinstance(item, dict) or item.get('domain', '').lstrip('.') not in DOMAINS
                    or not isinstance(item.get('name'), str) or not isinstance(item.get('value'), str)):
                raise ValueError('Invalid cached cookie')
            vpn.session.cookies.set_cookie(requests.cookies.create_cookie(**item))
    except (ValueError, KeyError, TypeError):
        if vpn is not None:
            vpn.session.close()
        print('[SessionCache] Unusable encrypted cache; fresh authentication required.', flush=True)
        return None
    try:
        response = vpn.session.get(get_vpn_url(config.ICOURSE_BASE + '/userapi/v1/infosimple'), timeout=30)
        if response.status_code in (401, 403):
            vpn.session.close()
            print('[SessionCache] Authorization expired; fresh authentication required.', flush=True)
            return None
        # Transport/server failures are not evidence of an expired login.
        response.raise_for_status()
        try:
            vpn._validate_icourse_response(response)
        except RuntimeError:
            print('[SessionCache] Session no longer authenticated; fresh authentication required.', flush=True)
            vpn.session.close()
            return None
        vpn.logged_in = True
        print('[SessionCache] Reused session; authenticated course API verified.', flush=True)
        return vpn
    except Exception:
        vpn.session.close()
        raise
