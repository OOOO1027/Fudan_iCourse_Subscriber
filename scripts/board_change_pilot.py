"""Full-lecture selection experiment. No production DB or schedule writes."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
from openai import OpenAI
from main import login_with_retry
from src.api.icourse import ICourseClient
from src.ai.board_selector import BoardSelector, REGIONS, crop
from src.ai.transcriber import Transcriber
from src.data.crypto_box import derive_new_password, encrypt, decrypt
from src.runtime import config
from board_vision_pilot import align_segments, image_request, media_command


def extract(client, directory):
    # Refresh the signed URL immediately before each independent read.
    url = client.get_video_url('38985', '693743')
    if not url:
        raise RuntimeError('Replay unavailable')
    stream, headers = client.get_stream_params(url)
    info = json.loads(media_command(['ffprobe','-v','error','-headers',headers,
        '-i',stream,'-show_entries','format=duration','-of','json']))
    duration = float(info['format']['duration'])
    if not 180 < duration < 21600:
        raise RuntimeError('Unexpected duration')
    url = client.get_video_url('38985', '693743')
    stream, headers = client.get_stream_params(url)
    cmd = ['ffmpeg','-nostdin','-y','-headers',headers,'-i',stream,
           '-map','0:v:0','-vf',"setpts=PTS-STARTPTS,select='isnan(prev_selected_t)+gte(t-prev_selected_t,5)',showinfo",
           '-fps_mode','vfr',str(directory/'sample-%05d.png'),
           '-map','0:a:0','-vn','-ar','16000','-ac','1','-c:a','pcm_s16le',str(directory/'audio.wav')]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    timer = threading.Timer(2700, proc.kill)
    timer.start()
    times = []
    try:
        for line in proc.stderr:
            match = re.search(rb'Parsed_showinfo.*\bn:\s*\d+.*pts_time:([\d.]+)', line)
            if match:
                times.append(float(match.group(1)))
                if len(times) % 120 == 0:
                    print('Extracted samples:', len(times), 'time:', times[-1], flush=True)
                    if shutil.disk_usage(directory).free < 2 * 1024**3:
                        proc.kill()
                        raise RuntimeError('Insufficient scratch space')
        code = proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        proc.stderr.close()
    if code != 0:
        raise RuntimeError('Video extraction failed; no production update')
    paths = sorted(directory.glob('sample-*.png'))
    if len(paths) != len(times) or not times or times[-1] < duration - 10:
        raise RuntimeError('Incomplete video coverage')
    return duration, list(zip(times, paths))


def main():
    password = derive_new_password(config.STUDENT_ID, config.PASSWORD)
    output = Path(os.environ['PILOT_OUTPUT'])
    output.mkdir(parents=True, exist_ok=True)
    report = {'schema':2, 'course_id':'38985','sub_id':'693743',
        'lecture':'2026-10-28第6-8节', 'status':'started', 'production_updated':False,
        'manual_review':'required', 'occlusion_method':'green-board visibility proxy, not person detection',
        'regions':REGIONS, 'events':[], 'frames':[], 'ocr_sampling_limit':12}
    manifest = []
    stage = 'login'

    def preserve(name, raw):
        encrypted = encrypt(raw, password)
        if decrypt(encrypted, password) != raw:
            raise RuntimeError('Encryption verification failed')
        (output/(name+'.enc')).write_bytes(encrypted)
        manifest.append({'file':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})

    with tempfile.TemporaryDirectory(prefix='board-change-') as temp:
        scratch = Path(temp)
        selected = {}
        try:
            client = ICourseClient(login_with_retry())
            course = client.get_course_detail('38985')
            if '自然语言处理' not in course['title']:
                raise RuntimeError('Course identity mismatch')
            stage = 'extract'
            duration, samples = extract(client, scratch)
            report['duration_s'] = duration
            report['sample_count'] = len(samples)
            report['coverage_end_s'] = samples[-1][0]

            def save(frame, timestamp, region, reason, detected_at, quality):
                stamp = round(timestamp*1000)
                report['events'].append({'timestamp_ms':stamp,'detected_ms':round(detected_at*1000),
                    'region':region,'reason':reason,'quality':quality})
                if stamp in selected:
                    return
                if len(selected) >= 800:
                    raise RuntimeError('Selection cap reached; thresholds need review')
                name = f'frame-{stamp:08d}.png'
                path = scratch/name
                Image.fromarray(frame).save(path)
                raw = path.read_bytes()
                preserve(name, raw)
                selected[stamp] = path
                report['frames'].append({'timestamp_ms':stamp,'file':name,
                    'sha256':hashlib.sha256(raw).hexdigest(),'width':frame.shape[1], 'height':frame.shape[0]})

            stage = 'select'
            selector = BoardSelector(save)
            for timestamp,path in samples:
                with Image.open(path) as image:
                    selector.add(np.array(image.convert('RGB')), timestamp)
            selector.finish()
            print('Full scan:', len(samples), 'samples; retained:', len(selected), 'originals', flush=True)
            stage = 'transcribe'
            reuse = os.environ.get('PILOT_REUSE_REPORT')
            if reuse:
                previous = json.loads(decrypt(Path(reuse).read_bytes(), password))
                if (previous['course_id'] != report['course_id'] or previous['sub_id'] != report['sub_id']
                        or abs(previous['duration_s']-duration) > 1 or not previous.get('segments')):
                    raise RuntimeError('Transcript cache identity mismatch')
                transcript, segments = previous['transcript'], previous['segments']
                report['transcript_source'] = 'verified previous full-lecture trial 41800144132'
                print('Reused existing full-lecture transcript:', len(segments), 'segments', flush=True)
            else:
                transcript, segments = Transcriber().transcribe_video(str(scratch/'audio.wav'))
            report['transcript'] = transcript
            report['segments'] = segments
            report['alignment'] = 'temporal_overlap_only_not_teacher_emphasis'
            for entry in report['frames']:
                entry['speech'] = align_segments(segments, entry['timestamp_ms'])
            preserve('transcript.json', json.dumps({'text':transcript,'segments':segments},ensure_ascii=False).encode())
            if not segments:
                raise RuntimeError('No speech; alignment cannot be accepted')
            api = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'],base_url='https://api.deepseek.com',
                         timeout=90,max_retries=0)
            ordered = sorted(selected)
            chosen = sorted(set(ordered[round(i*(len(ordered)-1)/11)] for i in range(12)))
            stage = 'vision'
            sheet = Image.new('RGB',(1280, len(chosen)*240),'white')
            for index, stamp in enumerate(chosen):
                path = selected[stamp]
                with Image.open(path) as image:
                    image = image.convert('RGB')
                    thumb = image.copy()
                    thumb.thumbnail((320,180))
                    sheet.paste(thumb,(0,index*240))
                    pixels = np.array(image)
                ImageDraw.Draw(sheet).text((5,index*240+222),f'{stamp/1000:.3f}s | full / left / center / right',fill='black')
                entry = next(x for x in report['frames'] if x['timestamp_ms']==stamp)
                entry['ocr_scope'] = 'left, center, right native crops; 12 retained timestamps sampled for review'
                entry['vision_review_status'] = 'unverified_do_not_use_as_summary_evidence'
                entry['vision'] = {}
                for col, (name, bounds) in enumerate(REGIONS.items(), start=1):
                    region_image = Image.fromarray(crop(pixels, bounds))
                    region_name = f'crop-{name}-{stamp:08d}.png'
                    region_path = scratch/region_name
                    region_image.save(region_path)
                    preserve(region_name, region_path.read_bytes())
                    zoom = region_image.copy()
                    zoom.thumbnail((310,215))
                    sheet.paste(zoom,(col*320,index*240))
                    try:
                        entry['vision'][name] = {'text':image_request(api,'deepseek-flash',region_path.read_bytes(),
                            '这是低清晰度课堂截图。只转录绿色黑板上确实能逐字看清的粉笔笔迹。'
                            '忽略投影幕、网页、讲台和衣服。擦痕不算笔迹。'
                            '看不清的每个词或公式位置写[无法辨认]，不要根据课程常识补全。'
                            '宁可少写也不要猜测。逐行记录，不解释、不改错。'
                            '没有笔迹则写没有可见板书。禁止推断老师强调或考试重点。')}
                    except Exception as exc:
                        entry['vision'][name] = {'error':type(exc).__name__}
                print('Crop OCR checked:', stamp, flush=True)
            preview = scratch/'contact-sheet.jpg'
            sheet.save(preview,quality=88)
            preserve(preview.name,preview.read_bytes())
            report['status'] = 'collected' if not any('error' in v for f in report['frames']
                for v in f.get('vision',{}).values()) else 'partial_vision_failure'
        except Exception as exc:
            report['status'] = 'failed'
            report['error_type'] = type(exc).__name__
            report['error_stage'] = stage
            print('Experiment failed:', stage, type(exc).__name__, flush=True)
        finally:
            preserve('report.json',json.dumps(report,ensure_ascii=False,indent=2).encode())
            (output/'manifest.json.enc').write_bytes(encrypt(json.dumps(manifest).encode(),password))
            print('Status:',report['status'],'saved originals:',len(selected),flush=True)
    return 0 if report['status']=='collected' else 1


if __name__ == '__main__': sys.exit(main())
