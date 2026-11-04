"""Offline visual coverage selection. Not semantic/OCR equivalence.

Keep the source archive intact. A reading set covers local chalk patterns;
uncovered patterns remain explicitly available for review.
"""
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageFilter
from src.ai.board_selector import REGIONS, crop

SIZE = 384
GRID = 4
PATCH = SIZE // GRID
POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


@dataclass
class BoardFeatures:
    ink: np.ndarray
    tolerant: np.ndarray
    visible: float


def extract_features(frame, regions=None, contrast=7):
    result = []
    for bounds in (regions or REGIONS).values():
        image = Image.fromarray(crop(frame, bounds)).resize((SIZE, SIZE), Image.Resampling.LANCZOS)
        hsv = np.asarray(image.convert('HSV'))
        gray = np.asarray(image.convert('L')).astype(np.float32)
        green = ((hsv[:,:,0] >= 42) & (hsv[:,:,0] <= 142)
                 & (hsv[:,:,1] > 15) & (hsv[:,:,2] < 205))
        mask = Image.fromarray(green.astype(np.uint8)*255)
        mask = mask.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(11))
        valid = np.asarray(mask) > 127
        background = np.asarray(image.convert('L').filter(ImageFilter.GaussianBlur(3))).astype(np.float32)
        # Local contrast suppresses broad lighting gradients and most erased smudges.
        ink = (gray-background > contrast) & valid & (gray < 205)
        ink[:4] = False
        ink[-4:] = False
        ink[:,:4] = False
        ink[:,-4:] = False
        tolerant = np.asarray(Image.fromarray(ink.astype(np.uint8)*255)
            .filter(ImageFilter.MaxFilter(3))) > 127
        result.append(BoardFeatures(ink, tolerant, float(valid.mean())))
    return result


def patches(array):
    return array.reshape(GRID,PATCH,GRID,PATCH).transpose(0,2,1,3).reshape(GRID*GRID,-1)


def coverage_atoms(features, threshold=.90, min_ink=35):
    """A source patch is covered only when its strokes occur in the target.

    Comparison is at the same board coordinates. Sliding can leave conservative
    duplicates; there is no guessed geometric/semantic equivalence.
    """
    n = len(features)
    all_cover, weights, origins = [], [], []
    for region in range(len(features[0])):
        source = np.stack([patches(f[region].ink) for f in features])
        target = np.stack([patches(f[region].tolerant) for f in features])
        for patch in range(GRID*GRID):
            counts = source[:,patch].sum(axis=1)
            active = np.flatnonzero(counts >= min_ink)
            if not len(active):
                continue
            packed_source = np.packbits(source[active,patch], axis=1)
            packed_target = np.packbits(target[:,patch], axis=1)
            cover = np.zeros((len(active),n),dtype=bool)
            for start in range(0,len(active),16):
                overlap = np.bitwise_and(packed_source[start:start+16,None,:], packed_target[None,:,:])
                hits = POPCOUNT[overlap].sum(axis=2)
                cover[start:start+16] = hits / counts[active[start:start+16],None] >= threshold
            # Each observation can always represent itself, even at strict settings.
            cover[np.arange(len(active)),active] = True
            # Stable material repeated in many frames should not dominate rare additions.
            weight = np.sqrt(counts[active].astype(float)) / cover.sum(axis=1)
            all_cover.append(cover)
            weights.append(weight)
            origins.extend((int(i),region,patch,int(counts[i])) for i in active)
    if not all_cover:
        return np.zeros((0,n),dtype=bool), np.zeros(0), []
    return np.vstack(all_cover), np.concatenate(weights), origins


def select_cover(cover, weights, quality=None, target=.99):
    if not 0 < target <= 1:
        raise ValueError('Coverage target outside (0,1]')
    n = cover.shape[1]
    quality = np.ones(n) if quality is None else np.asarray(quality)
    uncovered = np.ones(len(weights),dtype=bool)
    selected = []
    total = float(weights.sum())
    trace = []
    while total and float(weights[uncovered].sum()) > total*(1-target)+1e-10:
        gain = weights[uncovered] @ cover[uncovered].astype(float)
        score = gain * (.95 + .05*quality)
        score[selected] = -1
        best = int(np.argmax(score))
        if gain[best] <= 0:
            raise RuntimeError('Uncoverable observations')
        selected.append(best)
        trace.append({'index':best,'new_weight':float(gain[best])})
        uncovered &= ~cover[:,best]
    # Remove choices made redundant by later, more complete boards.
    for chosen in list(reversed(selected)):
        others = [i for i in selected if i != chosen]
        covered = cover[:,others].any(axis=1) if others else np.zeros(len(weights),dtype=bool)
        if total and float(weights[covered].sum()) >= target*total-1e-10:
            selected.remove(chosen)
    covered = cover[:,selected].any(axis=1) if selected else np.zeros(len(weights),dtype=bool)
    return selected, covered, trace


def curate(features, timestamps, threshold=.90, target=.99):
    if not features or len(features) != len(timestamps):
        raise ValueError('Need paired frames and timestamps')
    cover, weights, origins = coverage_atoms(features,threshold=threshold)
    quality = np.array([np.mean([b.visible for b in f]) for f in features])
    selected, covered, trace = select_cover(cover,weights,quality,target)
    strict, _, _ = select_cover(cover,weights,quality,1.0)
    records = []
    for index,timestamp in enumerate(timestamps):
        atoms = np.array([j for j,o in enumerate(origins) if o[0]==index],dtype=int)
        if len(atoms):
            local_weights = np.array([origins[a][3] for a in atoms],dtype=float)
            ratio = float(local_weights @ covered[atoms] / local_weights.sum())
            coverage_by_selected = [(int(s),float(local_weights @ cover[atoms,s] / local_weights.sum()))
                                    for s in selected]
            matches = sorted(coverage_by_selected,key=lambda x:x[1],reverse=True)[:3]
        else:
            ratio, matches = None, []
        status = ('selected' if index in selected else 'low_visual_evidence' if not len(atoms)
                  else 'review_uncovered' if ratio < .98 else 'covered_by_reading_set')
        records.append({'index':index,'timestamp_ms':timestamp,'status':status,
            'observed_patch_count':len(atoms),'stroke_weighted_patch_coverage':ratio,
            'best_representatives':[{'index':s,'coverage':v} for s,v in matches]})
    return {'selected_indices':sorted(selected,key=lambda i:timestamps[i]),
        'strict_selected_indices':sorted(strict,key=lambda i:timestamps[i]),
        'weighted_visual_coverage':float(weights[covered].sum()/weights.sum()) if weights.sum() else None,
        'patch_observations':len(origins),'uncovered_observations':int((~covered).sum()),
        'threshold':threshold,'target':target,'trace':trace,'frames':records,
        'metric_warning':'Visual patch coverage is not semantic information retention or OCR accuracy.'}
