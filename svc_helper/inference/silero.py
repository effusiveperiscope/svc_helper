from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import librosa
import soundfile as sf
import numpy as np

class SileroChunker:
    def __init__(self):
        self.model = load_silero_vad()
        self.silero_sr = 16000
    
    def __call__(self, 
        wav_true : np.ndarray, # usually 48khz or 44.1khz
        true_sr=48000,
        front_buffer=1, # seconds
        max_len=5 # seconds
        ):
        wav_silero = librosa.resample(wav_true,
            orig_sr=true_sr, 
            target_sr=self.silero_sr)
        speech_timestamps = get_speech_timestamps(
            wav_silero, self.model)

        sr = true_sr
        
        buffered_timestamps_sr = []
        # Stage 1. Front buffer timestamps and convert to true SR
        for i, timestamp in enumerate(speech_timestamps):
            start = timestamp['start'] / self.silero_sr
            end = timestamp['end'] / self.silero_sr

            front_samples = int(front_buffer * sr)
            start_samples = int(start * sr)
            end_samples = int(end * sr)

            if i == 0:
                start_samples = max(start_samples - front_samples, 0)
            else:
                start_samples = max(start_samples - front_samples, 
                    int(speech_timestamps[i - 1]['end'] / self.silero_sr * sr))

            buffered_timestamps_sr.append({
                'start': start_samples,
                'end': end_samples
            })

        # Stage 2. Split timestamps by max_len
        true_timestamps_sr = []
        for timestamp in buffered_timestamps_sr:
            start = timestamp['start']
            end = timestamp['end']

            while end - start > max_len * sr:
                true_timestamps_sr.append({
                    'start': start,
                    'end': start + max_len * sr
                })
                start = start + max_len * sr

            true_timestamps_sr.append({
                'start': start,
                'end': end
            })

        # Convert timestamps to audio output
        output = []
        for timestamp in true_timestamps_sr:
            output.append({
                'start': timestamp['start'],
                'end': timestamp['end'],
                'wav': wav_true[timestamp['start']:timestamp['end']]})

        return output