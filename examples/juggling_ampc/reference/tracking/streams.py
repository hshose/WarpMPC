"""Same-host, bounded subprocess streams; no network/robot side effects on import."""
from collections import deque
import json
from pathlib import Path
import subprocess
import threading
import time
import numpy as np
from .calibration import pose


class JsonProcess:
    def __init__(self, command, output, name, writable=False):
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        self.stderr = (output/(name+'.stderr')).open('x')
        self.log = (output/(name+'.jsonl')).open('x')
        self.command_log = (output/(name+'_commands.jsonl')).open('x') if writable else None
        self.process = subprocess.Popen(list(map(str, command)), stdin=subprocess.PIPE if writable else subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=self.stderr, text=True, bufsize=1)
        self.queue = deque(); self.lock = threading.Lock(); self.error = None; self.closed = False
        self.vicon_stream = name == 'vicon'
        self.vicon_previous = None
        self.thread = threading.Thread(target=self._read, daemon=True); self.thread.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                item = json.loads(line)
                if not isinstance(item, dict): raise ValueError('Expected JSON object')
                if self.vicon_stream:
                    # Preserve useful per-frame SDK latency: this recording's
                    # receive-latency clock agrees with frame cadence to <0.2 ms.
                    # Missing metadata bridges via the previous capture clock.
                    reported = usable_vicon_latency(item.get('latency_s'))
                    previous = getattr(self, 'vicon_previous', None)
                    received = float(item['receive_time'])
                    number = item['frame']; rate = float(item.get('frame_rate_hz') or 100.)
                    cadence = (previous[1]+(number-previous[0])/rate
                               if previous and number > previous[0] and rate > 0 else None)
                    stamp = received-reported if reported is not None else cadence
                    source = 'receive_minus_sdk_latency' if reported is not None else 'frame_cadence_fallback'
                    if stamp is None or stamp > received or (previous and number > previous[0] and stamp <= previous[1]):
                        stamp = cadence if cadence is not None and cadence <= received else received-.01
                        source = 'frame_cadence_fallback' if stamp == cadence else 'receive_minus_10ms_fallback'
                    self.vicon_previous = (number, stamp)
                    item['capture_time_s'] = stamp
                    item['effective_latency_s'] = received-stamp
                    item['timing_source'] = source
                    item['sdk_latency_usable'] = reported is not None
                    self.log.write(json.dumps(item)+'\n')
                else: self.log.write(line)
                with self.lock:
                    if len(self.queue) >= 2048: raise RuntimeError('Measurement consumer is stalled')
                    self.queue.append(item)
        except Exception as e: self.error = e
        finally: self.log.flush()

    def drain(self):
        if self.error: raise RuntimeError('Measurement stream failed') from self.error
        if self.process.poll() is not None: raise RuntimeError(f'{self.process.args[0]} exited ({self.process.returncode}); see stderr log')
        with self.lock: items = list(self.queue); self.queue.clear()
        return items

    def send(self, text):
        if self.process.poll() is not None: raise RuntimeError('Robot session exited')
        self.process.stdin.write(text+'\n'); self.process.stdin.flush()
        if self.command_log is not None and text != 'heartbeat':
            self.command_log.write(json.dumps(dict(send_time=time.monotonic(),command=text))+'\n')
            self.command_log.flush()

    def close(self):
        if self.closed: return
        self.closed = True
        if self.process.poll() is None:
            if self.process.stdin:
                try: self.send('stop')
                except (BrokenPipeError, RuntimeError): pass
            else: self.process.terminate()
            try: self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try: self.process.wait(timeout=2)
                except subprocess.TimeoutExpired: self.process.kill(); self.process.wait(timeout=2)
        self.thread.join(timeout=2)
        if self.process.stdin: self.process.stdin.close()
        self.process.stdout.close(); self.stderr.close(); self.log.close()
        if self.command_log is not None: self.command_log.close()


def usable_vicon_latency(value):
    try: value = float(value)
    except (TypeError, ValueError): return None
    return value if np.isfinite(value) and .001 <= value <= .05 else None


def capture_time(frame, extra_latency=0.):
    received = float(frame['receive_time'])
    if not np.isfinite(received): raise ValueError('Invalid Vicon receive timestamp')
    if not np.isfinite(extra_latency) or not -.01 <= extra_latency < .05:
        raise ValueError('Invalid configured extra latency')
    # A signed residual calibration aligns the approximate camera clock to
    # robot readback. Negative extra_latency advances camera timestamps.
    if 'capture_time_s' in frame:
        timestamp=float(frame['capture_time_s'])
        if np.isfinite(timestamp) and timestamp <= received: return timestamp-extra_latency
    latency = usable_vicon_latency(frame.get('latency_s'))
    return received-(.01 if latency is None else latency)-extra_latency


def paddle_pose(frame):
    p = frame.get('paddle')
    return None if p is None else pose(p['position_m'], p['quaternion_xyzw'])


def ball_position(frame, selector='hsh_pingpong_ball'):
    """Select Subject/Marker, globally unique marker, or one-marker subject.

    A single marker has no orientation. Never choose an arbitrary unlabeled dot
    or the origin of an unrelated rigid body. Ambiguity is a configuration error.
    """
    markers = frame.get('markers', [])
    if '/' in selector:
        subject, name = selector.split('/', 1)
        matches = [m for m in markers if m['subject'] == subject and m['name'] == name]
    else:
        matches = [m for m in markers if m['name'] == selector]
        if not matches: matches = [m for m in markers if m['subject'] == selector]
    if len(matches) > 1: raise ValueError('Ambiguous ball marker; specify Subject/Marker explicitly')
    if not matches or matches[0]['position_m'] is None: return None
    p = np.array(matches[0]['position_m'], float)
    if p.shape != (3,) or not np.isfinite(p).all(): raise ValueError('Invalid ball marker position')
    return p
