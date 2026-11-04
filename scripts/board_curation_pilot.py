"""Curate the saved 266 frames offline. No video, school login, ASR or model API."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.ai.board_curator import curate, extract_features
from src.data.crypto_box import derive_new_password, decrypt, encrypt

SOURCE_ID = '41802975078-1'


def time_label(ms):
    sec=ms/1000
    return f'{int(sec//3600):02d}:{int(sec//60)%60:02d}:{sec%60:06.3f}'


def main():
    source=Path(os.environ['CURATION_SOURCE'])/'evidence'/SOURCE_ID
    output=Path(os.environ['CURATION_OUTPUT'])
    output.mkdir(parents=True,exist_ok=True)
    password=derive_new_password(os.environ['StuId'],os.environ['UISPsw'])
    manifest=json.loads(decrypt((source/'manifest.json.enc').read_bytes(),password))
    source_hashes={x['file']:x['sha256'] for x in manifest}
    def read(name):
        if Path(name).name != name:
            raise ValueError('Unexpected source filename')
        raw=decrypt((source/(name+'.enc')).read_bytes(),password)
        if hashlib.sha256(raw).hexdigest()!=source_hashes[name]:
            raise ValueError('Source integrity check failed')
        return raw
    previous=json.loads(read('report.json'))
    frames=sorted(previous['frames'],key=lambda f:f['timestamp_ms'])
    if len(frames)!=266 or previous['sub_id']!='693743':
        raise ValueError('Source dataset mismatch')
    results=[]
    def preserve(name,raw):
        encrypted=encrypt(raw,password)
        if decrypt(encrypted,password)!=raw:
            raise ValueError('Output integrity check failed')
        (output/(name+'.enc')).write_bytes(encrypted)
        results.append({'file':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
    with tempfile.TemporaryDirectory(prefix='board-curation-') as temp:
        scratch=Path(temp)
        features=[]
        for index,frame in enumerate(frames):
            raw=read(frame['file'])
            (scratch/frame['file']).write_bytes(raw)
            with Image.open(io.BytesIO(raw)) as im:
                features.append(extract_features(np.array(im.convert('RGB'))))
            if (index+1)%50==0:
                print('Verified saved frames:',index+1,flush=True)
        times=[f['timestamp_ms'] for f in frames]
        report=curate(features,times)
        report.update(source_run=SOURCE_ID,source_branch='adaptive-board-evidence',
            course_id=previous['course_id'],sub_id=previous['sub_id'],source_count=len(frames),
            source_unchanged=True,production_updated=False,uses_model_api=False,
            manual_review='required',semantic_retention='not_measured')
        for record,frame in zip(report['frames'],frames):
            record.update(file=frame['file'],sha256=frame['sha256'])
        report['sensitivity']=[]
        for threshold in (.85,.95):
            alternative=curate(features,times,threshold=threshold)
            report['sensitivity'].append({'threshold':threshold,
                'selected':len(alternative['selected_indices']),
                'strict_selected':len(alternative['strict_selected_indices']),
                'visual_coverage':alternative['weighted_visual_coverage']})
        selected=report['selected_indices']
        for i in selected:
            frame=frames[i]
            shutil.copyfile(source/(frame['file']+'.enc'),output/(frame['file']+'.enc'))
            results.append({'file':frame['file'],'sha256':frame['sha256'],
                'bytes':(scratch/frame['file']).stat().st_size})
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
        def board_thumb(i,width=440,height=150):
            with Image.open(scratch/frames[i]['file']) as image:
                board=image.crop((40,200,1890,815)).convert('RGB')
            board.thumbnail((width,height))
            return board
        for page,start in enumerate(range(0,len(selected),12),start=1):
            batch=selected[start:start+12]
            sheet=Image.new('RGB',(1760,((len(batch)+3)//4)*180),'white')
            draw=ImageDraw.Draw(sheet)
            for position,i in enumerate(batch):
                x,y=(position%4)*440,(position//4)*180
                sheet.paste(board_thumb(i),(x,y))
                draw.text((x+5,y+153),f'{start+position+1:02d}  {time_label(times[i])}',fill='black',font=font)
            buf=io.BytesIO();sheet.save(buf,format='JPEG',quality=90)
            preserve(f'reading-{page:02d}.jpg',buf.getvalue())
        # Trace actual skipped frames to their best representatives, not invented examples.
        eligible=[r for r in report['frames'] if r['status']=='covered_by_reading_set'
                  and r['best_representatives'] and r['best_representatives'][0]['coverage']>=.95
                  and r['observed_patch_count']>=5]
        eligible.sort(key=lambda r:r['timestamp_ms'])
        sample=[]
        if eligible:
            for index in np.linspace(0,len(eligible)-1,min(6,len(eligible))).astype(int):
                record=eligible[index]
                sample.append((record['index'],record['best_representatives'][0]['index'],
                               record['best_representatives'][0]['coverage']))
        sheet=Image.new('RGB',(1280,max(1,len(sample))*250),'white')
        draw=ImageDraw.Draw(sheet)
        for row,(old,new,coverage) in enumerate(sample):
            for col,i in enumerate((old,new)):
                sheet.paste(board_thumb(i,630,215),(col*640,row*250))
                label=('Skipped' if col==0 else 'Kept')+' '+time_label(times[i])
                draw.text((col*640+5,row*250+218),label,fill='black',font=font)
        buf=io.BytesIO();sheet.save(buf,format='JPEG',quality=90)
        preserve('replacement-pairs.jpg',buf.getvalue())
        report['example_pairs']=[{'skipped_index':a,'selected_index':b,'coverage':c} for a,b,c in sample]
        # Expose low-evidence inputs and residual patterns as review material.
        review=[r['index'] for r in report['frames'] if r['status'] in ('review_uncovered','low_visual_evidence')]
        report['review_indices']=review
        for page,start in enumerate(range(0,len(review),12),start=1):
            batch=review[start:start+12]
            sheet=Image.new('RGB',(1760,((len(batch)+3)//4)*180),'white')
            draw=ImageDraw.Draw(sheet)
            for pos,i in enumerate(batch):
                x,y=pos%4*440,pos//4*180
                sheet.paste(board_thumb(i),(x,y))
                draw.text((x+5,y+153),time_label(times[i]),fill='black',font=font)
            buf=io.BytesIO();sheet.save(buf,format='JPEG',quality=88)
            preserve(f'review-{page:02d}.jpg',buf.getvalue())
        preserve('report.json',json.dumps(report,ensure_ascii=False,indent=2).encode())
        (output/'manifest.json.enc').write_bytes(encrypt(json.dumps(results).encode(),password))
        print(json.dumps({'source_frames':len(frames),'reading_frames':len(selected),
            'strict_reading_frames':len(report['strict_selected_indices']),
            'visual_coverage':report['weighted_visual_coverage'],
            'review_frames':len(review),'sensitivity':report['sensitivity']}),flush=True)


if __name__=='__main__': main()
