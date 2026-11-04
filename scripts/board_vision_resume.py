"""Retry only missing vision results from an encrypted pilot checkpoint."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

from board_vision_pilot import image_request
from openai import OpenAI
from src.data.crypto_box import derive_new_password, encrypt, decrypt


def main():
    source = Path(os.environ['PILOT_SOURCE'])
    output = Path(os.environ['PILOT_OUTPUT'])
    output.mkdir(parents=True, exist_ok=True)
    password = derive_new_password(os.environ['StuId'], os.environ['UISPsw'])
    manifest = json.loads(decrypt((source / 'manifest.json.enc').read_bytes(), password))
    payloads = {}
    for entry in manifest:
        name = entry['file']
        if Path(name).name != name:
            raise ValueError('Invalid evidence name')
        blob = (source / (name + '.enc')).read_bytes()
        raw = decrypt(blob, password)
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Evidence integrity check failed')
        payloads[name] = raw
        (output / (name + '.enc')).write_bytes(blob)
    report = json.loads(payloads['report.json'])
    print('Previous vision errors:', dict(Counter(
        x.get('error_type', 'none') for x in report['frames'])), flush=True)
    api = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'],
                 base_url='https://api.deepseek.com', timeout=90, max_retries=0)
    try:
        for entry in report['frames']:
            if entry.get('vision_text'):
                continue
            try:
                text = image_request(api, 'deepseek-flash', payloads[entry['file']],
                    '只依据图片转录。将真实黑板/白板手写、投影PPT、屏幕手写标注分别列出。'
                    '没有板书就明确写没有可见板书；无法辨认的字和公式标记[无法辨认]，'
                    '不要凭常识补全，不要推断老师强调或考试重点。描述板书所在区域。'
                    '不要把投影文字当作黑板板书。用中文回答。')
                entry['vision_text'] = text
                entry['model'] = 'deepseek-flash'
                entry.pop('error_type', None)
                entry.pop('http_status', None)
                print('Vision completed:', entry['timestamp_ms'], 'chars:', len(text), flush=True)
            except Exception as exc:
                entry['error_type'] = type(exc).__name__
                entry['http_status'] = getattr(exc, 'status_code', None)
                print('Vision failed:', entry['timestamp_ms'], type(exc).__name__,
                      getattr(exc, 'status_code', None), flush=True)
    finally:
        completed = sum(bool(x.get('vision_text')) for x in report['frames'])
        report['status'] = 'collected' if completed == len(report['frames']) else 'partial_vision_failure'
        report['resumed_from'] = os.environ['PILOT_RESUME']
        raw = json.dumps(report, ensure_ascii=False, indent=2).encode()
        blob = encrypt(raw, password)
        assert decrypt(blob, password) == raw
        (output / 'report.json.enc').write_bytes(blob)
        for entry in manifest:
            if entry['file'] == 'report.json':
                entry.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        (output / 'manifest.json.enc').write_bytes(encrypt(json.dumps(manifest).encode(), password))
        print('Completed vision results:', completed, '/', len(report['frames']), flush=True)
        print('Original frames and audio: preserved byte-for-byte; manual review still required', flush=True)
    return 0 if report['status'] == 'collected' else 1


if __name__ == '__main__':
    sys.exit(main())
