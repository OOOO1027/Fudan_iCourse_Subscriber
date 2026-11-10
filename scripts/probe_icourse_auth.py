"""Bounded cloud authentication probe; no course/database writes or secrets logged."""
import json
import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.api.webvpn import WebVPNSession
from src.api.icourse import ICourseClient
from src.runtime import config


def shape(value, depth=0):
    if depth > 3:
        return type(value).__name__
    if isinstance(value, dict):
        return {k: shape(v, depth+1) for k, v in value.items()
                if re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]{0,50}', k)}
    if isinstance(value, list):
        return [shape(v, depth+1) for v in value[:2]]
    if value is None or isinstance(value, bool):
        return value
    return type(value).__name__


def describe(data):
    result = {'shape': shape(data)}
    if isinstance(data, dict):
        for key in ('code', 'status', 'errorCode'):
            if re.fullmatch(r'-?\d{1,6}', str(data.get(key, ''))):
                result[key] = data[key]
        messages = ' '.join(str(data.get(k, '')) for k in ('msg', 'message', 'errorMsg')).lower()
        result['signals'] = [name for name, words in {
            'captcha': ('验证码', 'captcha', 'verifycode'),
            'expired': ('过期', '失效', 'expired', 'timeout'),
            'password': ('密码', 'password'),
            'risk': ('风险', 'risk'),
            'locked': ('锁定', 'locked'),
            'success': ('成功', 'success'),
        }.items() if any(w in messages for w in words)]
    return result


def probe(warm_page=False):
    vpn = WebVPNSession()
    try:
        vpn.login()
        get, post = vpn.session.get, vpn.session.post
        context = {}

        def observed_get(url, **kwargs):
            response = get(url, **kwargs)
            location = response.headers.get('Location', '')
            if re.search(r'[?&]lck=', location):
                context['page'] = urljoin(response.url, location)
            return response

        def observed_post(url, **kwargs):
            endpoint = urlparse(url).path.rsplit('/', 1)[-1]
            if warm_page and endpoint == 'queryAuthMethods' and context.get('page'):
                page = context.pop('page')
                if urlparse(page).hostname not in {'webvpn.fudan.edu.cn', 'id.fudan.edu.cn'}:
                    raise RuntimeError('Untrusted context host')
                res = get(page, timeout=60)
                print('[Probe] context page status:', res.status_code, flush=True)
            response = post(url, **kwargs)
            if endpoint in {'authExecute', 'queryAuthMethods'}:
                try:
                    details = describe(response.json())
                except ValueError:
                    details = {'format': 'not-json'}
                print('[Probe]', endpoint, 'HTTP', response.status_code,
                      json.dumps(details), flush=True)
            return response

        vpn.session.get, vpn.session.post = observed_get, observed_post
        vpn.authenticate_icourse()
        client = ICourseClient(vpn)
        if not client.check_alive():
            raise RuntimeError('Authenticated API probe failed')
        for course_id in config.COURSE_IDS:
            detail = client.get_course_detail(course_id)
            lectures = detail['lectures']
            print('[Probe] course', course_id, 'lectures', len(lectures),
                  'playbacks', sum(bool(x.get('has_playback')) for x in lectures), flush=True)
        return True
    except Exception as exc:
        # Existing auth errors can contain redirect URLs; report only the class.
        print('[Probe] failed:', type(exc).__name__, flush=True)
        return False
    finally:
        vpn.session.close()


if __name__ == '__main__':
    print('[Probe] Existing flow', flush=True)
    if not probe():
        print('[Probe] Follow context page in a fresh session', flush=True)
        if not probe(warm_page=True):
            sys.exit(1)
