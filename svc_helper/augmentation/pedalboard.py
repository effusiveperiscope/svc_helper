import pedalboard
import random
import numpy as np

def _randp(prob):
    return (random.random() < prob)

# Randomly applies various pedalboard effects to audio
class PedalboardRandomAugmentor:
    def __init__(self, probs : dict = {}):
        """
        Accepts a dictionary of probabilities for various augmentations (see code for details)
        """
        default_probs = {
            'base': 0.3, # probability of activating augmentation
            'clipping': 0.05,
            'comp_gentle': 0.2,
            'comp_hard': 0.2,
            'quiet': 0.2,
            'limit': 0.1,
            'resample_8k': 0.05,
            'resample_16k': 0.05,
            'resample_22k': 0.05,
            'resample_24k': 0.05,
            'reverb_short': 0.05,
            'reverb_long': 0.05,
            'delay_short': 0.05,
            'bitcrush_8': 0.1,
            'mp3_vbr2': 0.1,
            'mp3_vbr0': 0.1,
        }
        self.probs = default_probs
        self.probs.update(probs)
        self.clipper = pedalboard.Clipping()
        self.comp_gentle = pedalboard.Compressor(
            threshold_db=-6.0, ratio=1.5, attack_ms=60.0)
        self.comp_hard = pedalboard.Compressor(
            threshold_db=-24.0, ratio=3)
        self.limit = pedalboard.Limiter(threshold_db=-6.0)
        self.quiet = pedalboard.Gain(gain_db=-6.0)
        self.resample_8k = pedalboard.Resample(target_sample_rate=8000.0,
            quality=pedalboard.Resample.Quality.Linear)
        self.resample_16k = pedalboard.Resample(target_sample_rate=16000.0,
            quality=pedalboard.Resample.Quality.Linear)
        self.resample_22k = pedalboard.Resample(target_sample_rate=22050.0,
            quality=pedalboard.Resample.Quality.Linear)
        self.resample_24k = pedalboard.Resample(target_sample_rate=24000.0,
            quality=pedalboard.Resample.Quality.Linear)
        self.reverb_short = pedalboard.Reverb(room_size = 0.1, wet_level = 0.1)
        self.reverb_long = pedalboard.Reverb(room_size = 0.9, wet_level = 0.3)
        self.delay_short = pedalboard.Delay(delay_seconds=0.05, feedback=0.3)
        self.bitcrush_8 = pedalboard.Bitcrush(bit_depth=8)
        self.mp3_vbr2 = pedalboard.MP3Compressor(vbr_quality=2.0)
        self.mp3_vbr0 = pedalboard.MP3Compressor(vbr_quality=0.1)

    def process(self, audio : np.ndarray, sr : float):
        self.clipper.reset()
        self.comp_gentle.reset()
        self.comp_hard.reset()
        self.limit.reset()
        self.delay_short.reset()
        self.reverb_short.reset()
        self.reverb_long.reset()

        if not _randp(self.probs['base']):
            return audio

        if _randp(self.probs['reverb_short']):
            audio = self.reverb_short.process(audio, sr)
        if _randp(self.probs['reverb_long']):
            audio = self.reverb_long.process(audio, sr)

        if _randp(self.probs['delay_short']):
            audio = self.delay_short.process(audio, sr)

        if _randp(self.probs['clipping']):
            audio = self.clipper.process(audio, sr)
        if _randp(self.probs['comp_gentle']):
            audio = self.comp_gentle.process(audio, sr)
        if _randp(self.probs['comp_hard']):
            audio = self.comp_hard.process(audio, sr)
        if _randp(self.probs['limit']):
            audio = self.limit.process(audio, sr)

        if _randp(self.probs['resample_8k']):
            audio = self.resample_8k.process(audio, sr)
        elif _randp(self.probs['resample_16k']):
            audio = self.resample_16k.process(audio, sr)
        elif _randp(self.probs['resample_22k']):
            audio = self.resample_22k.process(audio, sr)
        elif _randp(self.probs['resample_24k']):
            audio = self.resample_24k.process(audio, sr)

        if _randp(self.probs['bitcrush_8']):
            audio = self.bitcrush_8.process(audio, sr)

        if _randp(self.probs['mp3_vbr2']) and sr >= 32000 and sr <= 48000:
            audio = self.mp3_vbr2.process(audio, sr)
        if _randp(self.probs['mp3_vbr0']) and sr >= 32000 and sr <= 48000:
            audio = self.mp3_vbr0.process(audio, sr)
        if _randp(self.probs['quiet']):
            audio = self.quiet.process(audio, sr)

        return audio