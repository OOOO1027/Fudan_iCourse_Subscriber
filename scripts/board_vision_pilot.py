"""Bounded, read-only course experiment; never updates the production database."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def align_segments(segments, timestamp_ms, radius_ms=15000):
    return [s for s in segments if s['end_ms'] >= timestamp_ms - radius_ms
            and s['start_ms'] <= timestamp_ms + radius_ms]


def shift_segments(segments, offset_ms):
    return [dict(s, start_ms=s['start_ms'] + offset_ms,
                 end_ms=s['end_ms'] + offset_ms) for s in segments]


def media_command(args, timeout=180):
    # ffmpeg errors can contain signed URLs and cookies: do not publish stderr.
    if args[0] == 'ffmpeg':
        args = [args[0], '-y', *args[1:]]
    for attempt in range(3):
        result = subprocess.run(args, capture_output=True, timeout=timeout)
        if not result.returncode:
            return result.stdout
        error = result.stderr.decode(errors='replace').lower()
        categories = [label for label in (
            '403 forbidden', '404 not found', '502 bad gateway', '503 service unavailable',
            '504 gateway', 'connection timed out', 'connection reset', 'end of file',
            'invalid data', 'moov atom not found', 'input/output error', 'permission denied'
        ) if label in error]
        print('Media failure:', args[0], 'attempt', attempt + 1,
              'exit', result.returncode, 'categories', categories, flush=True)
        if attempt < 2:
            time.sleep(5)
    raise RuntimeError('Media extraction failed (details withheld from public logs)')


def image_request(client, model, image_bytes, prompt):
    result = client.chat.completions.create(
        model=model, max_tokens=1600,
        messages=[{'role': 'user', 'content': [
            {'type': 'text', 'text': prompt},
            {'type': 'image_url', 'image_url': {
                'url': 'data:image/png;base64,' + base64.b64encode(image_bytes).decode(),
                'detail': 'high'}}]}])
    if result.choices[0].finish_reason != 'stop':
        raise RuntimeError('Incomplete model response')
    return result.choices[0].message.content or ''


def main():
    from PIL import Image, ImageDraw, ImageFont
    from openai import OpenAI
    from src.runtime import config
    from src.api.icourse import ICourseClient
    from src.ai.transcriber import Transcriber
    from src.data.crypto_box import derive_new_password, encrypt, decrypt
    from main import login_with_retry

    for key in ('StuId', 'UISPsw', 'DEEPSEEK_API_KEY', 'COURSE_IDS'):
        if not os.environ.get(key):
            raise RuntimeError('Missing required environment: ' + key)
    root = Path(os.environ['PILOT_OUTPUT'])
    root.mkdir(parents=True, exist_ok=True)
    report = {'schema': 1, 'status': 'started', 'production_updated': False,
              'review_status': 'manual_review_required', 'frames': [], 'clips': []}
    api = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'],
                 base_url=os.environ.get('DEEPSEEK_BASE_URL') or 'https://api.deepseek.com',
                 timeout=90, max_retries=0)
    password = derive_new_password(config.STUDENT_ID, config.PASSWORD)
    with tempfile.TemporaryDirectory(prefix='board-pilot-') as temp:
        evidence = Path(temp) / 'evidence'
        evidence.mkdir()
        try:
            # A random visual challenge checks actual image reading, not model self-report.
            code = secrets.token_hex(4).upper()
            image = Image.new('RGB', (800, 180), 'white')
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 64)
            ImageDraw.Draw(image).text((40, 40), code, fill='black', font=font)
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            (evidence / 'capability.png').write_bytes(buffer.getvalue())
            report['capability'] = {}
            for model in ('deepseek-v4-pro', 'deepseek-flash'):
                try:
                    answer = image_request(api, model, buffer.getvalue(),
                                           'Return only the characters visible in this image.')
                    passed = answer.strip() == code
                    report['capability'][model] = {'passed': passed, 'response': answer}
                except Exception as exc:
                    report['capability'][model] = {
                        'passed': False, 'error_type': type(exc).__name__,
                        'http_status': getattr(exc, 'status_code', None)}
                print('Image capability:', model, report['capability'][model]['passed'], flush=True)
            if not report['capability']['deepseek-flash']['passed']:
                raise RuntimeError('Vision capability probe did not pass')

            vpn = login_with_retry()
            client = ICourseClient(vpn)
            matches = []
            for course_id in config.COURSE_IDS:
                detail = client.get_course_detail(course_id)
                if ('自然语言处理' in detail['title'] and '周宝健' in detail['teacher']):
                    matches.append((course_id, detail))
            if len(matches) != 1:
                raise RuntimeError('Expected one subscribed NLP course')
            course_id, detail = matches[0]
            report['course_id'] = course_id
            report['course_title'] = detail['title']
            lectures = sorted(detail['lectures'], key=lambda item: (item['date'], str(item['sub_id'])), reverse=True)
            report['catalog'] = lectures
            chosen = None
            for lecture in lectures:
                if not lecture['has_playback']:
                    continue
                url = client.get_video_url(course_id, str(lecture['sub_id']))
                if url:
                    chosen = lecture
                    break
            if chosen is None:
                raise RuntimeError('No downloadable replay found')
            report['lecture'] = chosen
            print('Selected lecture:', course_id, chosen['sub_id'], chosen['sub_title'], flush=True)
            stream, headers = client.get_stream_params(url)
            duration = float(media_command([
                'ffprobe', '-v', 'error', '-headers', headers, '-i', stream,
                '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1'
            ]).decode().strip())
            if not 180 <= duration <= 21600:
                raise RuntimeError('Unexpected video duration')
            report['duration_s'] = duration
            # Three separate windows are a pilot, not full lecture coverage.
            starts = [int((duration - 90) * ratio) for ratio in (0.15, 0.5, 0.85)]
            transcriber = Transcriber()
            for clip_index, start in enumerate(starts):
                audio_path = evidence / f'audio-{clip_index}.wav'
                media_command(['ffmpeg', '-nostdin', '-v', 'error', '-headers', headers,
                               '-ss', str(start), '-i', stream, '-t', '90', '-vn',
                               '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(audio_path)], 300)
                text, segments = transcriber.transcribe_video(str(audio_path))
                shifted = shift_segments(segments, start * 1000)
                report['clips'].append({'start_ms': start * 1000, 'end_ms': (start + 90) * 1000,
                                        'audio': audio_path.name, 'transcript': text, 'segments': shifted})
                for offset in (0, 30, 60, 89):
                    timestamp = start + offset
                    frame_path = evidence / f'frame-{timestamp:06d}.png'
                    # Native decoded resolution, lossless PNG; no crop, resize, or annotation.
                    media_command(['ffmpeg', '-nostdin', '-v', 'error', '-headers', headers,
                                   '-ss', str(timestamp), '-i', stream, '-frames:v', '1',
                                   str(frame_path)], 180)
                    raw = frame_path.read_bytes()
                    with Image.open(frame_path) as frame:
                        width, height = frame.size
                    entry = {'timestamp_ms': timestamp * 1000, 'file': frame_path.name,
                             'sha256': hashlib.sha256(raw).hexdigest(),
                             'width': width, 'height': height,
                             'speech_window_ms': 15000,
                             'speech': align_segments(shifted, timestamp * 1000),
                             'alignment': 'temporal_overlap_only_not_semantic_confirmation'}
                    report['frames'].append(entry)
                    try:
                        entry['vision_text'] = image_request(api, 'deepseek-flash', raw,
                            '只依据图片转录。将真实黑板/白板手写、投影PPT、屏幕手写标注分别列出。'
                            '没有板书就明确写没有可见板书；无法辨认的字和公式标记[无法辨认]，'
                            '不要凭常识补全，不要推断老师强调或考试重点。描述板书所在区域。'
                            '不要把投影文字当作黑板板书。用中文回答。')
                        entry['model'] = 'deepseek-flash'
                    except Exception as exc:
                        entry['error_type'] = type(exc).__name__
                        entry['http_status'] = getattr(exc, 'status_code', None)
                    print('Frame saved and processed:', timestamp, width, height, flush=True)
            report['status'] = ('collected' if all('vision_text' in x for x in report['frames'])
                                else 'partial_vision_failure')
        except Exception as exc:
            report['status'] = 'failed'
            report['error_type'] = type(exc).__name__
            # Never log exception bodies: upstream errors may include access URLs.
            print('Pilot stopped:', type(exc).__name__, flush=True)
        finally:
            (evidence / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
            manifest = []
            for path in sorted(evidence.iterdir()):
                raw = path.read_bytes()
                encrypted = encrypt(raw, password)
                if decrypt(encrypted, password) != raw:
                    raise RuntimeError('Evidence encryption round-trip mismatch')
                (root / (path.name + '.enc')).write_bytes(encrypted)
                manifest.append({'file': path.name, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
            (root / 'manifest.json.enc').write_bytes(encrypt(json.dumps(manifest).encode(), password))
            print('Encrypted evidence files:', len(manifest), 'status:', report['status'], flush=True)
    return 0 if report['status'] == 'collected' else 1


if __name__ == '__main__':
    sys.exit(main())
