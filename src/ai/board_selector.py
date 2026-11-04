"""Streaming classroom-specific keyframe selection, not semantic event detection."""
from dataclasses import dataclass
from PIL import Image, ImageFilter
import numpy as np

# Calibrated against the three original 1920x1080 pilot frames, not other rooms.
REGIONS = {'left': (.025, .19, .33, .74),
           'center': (.355, .19, .655, .75),
           'right': (.685, .20, .98, .75)}


def crop(frame, region):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = region
    return frame[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]


def features(frame, region):
    small = Image.fromarray(crop(frame, region)).resize((240, 180))
    gray = small.convert('L')
    hsv = np.asarray(small.convert('HSV'))
    green = ((hsv[:, :, 0] >= 42) & (hsv[:, :, 0] <= 142)
             & (hsv[:, :, 1] > 15) & (hsv[:, :, 2] < 210))
    # Close small chalk gaps, but exclude large non-board foreground objects.
    mask = np.asarray(Image.fromarray(green.astype(np.uint8)*255)
        .filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(9))) > 127
    blurred = np.asarray(gray.filter(ImageFilter.GaussianBlur(.8)))
    edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES))[2:-2,2:-2]
    visible = float(green.mean())
    sharpness = float(edges.var())
    return {'gray': blurred.astype(np.float32) / 255, 'mask': mask,
            'ink': float((edges > 24).mean()), 'visible': visible,
            'quality': visible + min(sharpness / 1000, .2)}


def distance(current, previous):
    if (current['mask'] & ~previous['mask']).mean() > .20:
        return 1.0
    valid = current['mask'] & previous['mask']
    if valid.mean() < .25:
        # A projection screen can initially hide the whole board. Allow a new
        # visible board to establish a baseline after two stable observations.
        return 1.0 if current['mask'].mean() >= .25 else 0.0
    delta = (current['gray'] - previous['gray'])[valid]
    # Remove uniform exposure shifts before judging a persistent board change.
    return float(np.abs(delta - np.median(delta)).mean())


@dataclass
class Candidate:
    time: float
    frame: object
    feature: dict


class BoardSelector:
    def __init__(self, save, regions=None, change=.012, stable=.025,
                 heartbeat=120, cooldown=30):
        self.save = save
        self.regions = regions or REGIONS
        self.change, self.stable = change, stable
        self.heartbeat, self.cooldown = heartbeat, cooldown
        self.states = {}
        self.sample_count = 0

    def emit(self, name, candidate, reason, detected_at):
        self.save(candidate.frame, candidate.time, name, reason, detected_at,
                  {k: v for k, v in candidate.feature.items() if k not in ('gray', 'mask')})

    def add(self, frame, timestamp):
        self.sample_count += 1
        for name, region in self.regions.items():
            f = features(frame, region)
            current = Candidate(timestamp, frame, f)
            if name not in self.states:
                self.states[name] = {'base': f, 'previous': f, 'best': current,
                                     'last_save': timestamp, 'last_change': timestamp,
                                     'pending': None}
                self.emit(name, current, 'initial', timestamp)
                continue
            state = self.states[name]
            delta = distance(f, state['base'])
            motion = distance(f, state['previous'])
            # Visible green area is only an occlusion proxy. Never label it a person detector.
            usable = f['visible'] >= max(.15, state['base']['visible'] - .12)
            changed = delta > self.change and usable
            if changed and timestamp - state['last_change'] >= self.cooldown:
                pending = state['pending']
                if pending is not None and motion < self.stable:
                    best = max((pending, current), key=lambda c: c.feature['quality'])
                    self.emit(name, state['best'], 'before_change', timestamp)
                    reason = ('possible_erasure_or_board_move'
                              if best.feature['ink'] < state['base']['ink'] * .75
                              else 'persistent_visual_change')
                    self.emit(name, best, reason, timestamp)
                    state.update(base=best.feature, best=best, last_change=timestamp,
                                 last_save=timestamp, pending=None)
                else:
                    state['pending'] = current
            else:
                state['pending'] = None
                if delta <= self.change and usable and motion < self.stable:
                    if current.feature['quality'] > state['best'].feature['quality']:
                        state['best'] = current
            if timestamp - state['last_save'] >= self.heartbeat:
                # Save the current scene even when blocked; do not silently reuse an old timestamp.
                self.emit(name, current, 'heartbeat', timestamp)
                state['last_save'] = timestamp
            state['previous'] = f

    def finish(self):
        for name, state in self.states.items():
            self.emit(name, state['best'], 'final_best', state['best'].time)
            if state['pending']:
                self.emit(name, state['pending'], 'unconfirmed_final_change', state['pending'].time)
