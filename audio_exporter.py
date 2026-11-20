import os
import time
import numpy as np
import soundfile as sf
import lameenc


def export_audio_file(samples: np.ndarray, sample_rate: int, out_dir: str, fmt: str) -> str:
    timestamp = int(time.time())
    base_name = f"docu_narration_{timestamp}"

    if fmt in ("wav", "flac", "ogg"):
        out_path = os.path.join(out_dir, f"{base_name}.{fmt}")
        sf.write(out_path, samples, sample_rate, format=fmt.upper())
        return out_path

    elif fmt == "mp3":
        out_path = os.path.join(out_dir, f"{base_name}.mp3")
        pcm_data = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(320)
        encoder.set_in_sample_rate(sample_rate)
        encoder.set_channels(1)
        encoder.set_quality(2)

        mp3_data = encoder.encode(pcm_data) + encoder.flush()
        with open(out_path, "wb") as f:
            f.write(mp3_data)
        return out_path

    raise ValueError(f"Unsupported format: {fmt}")
