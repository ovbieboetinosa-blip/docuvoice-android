import os
import re
import threading
import urllib.request
import tarfile
import numpy as np
import sherpa_onnx

from kivy.app import App
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.uix.progressbar import ProgressBar

from audio_exporter import export_audio_file

VOICES = {
    "Marcus (Deep US Male - History)": {
        "folder": "vits-piper-en_US-ryan-high",
        "model": "en_US-ryan-high.onnx",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-ryan-high.tar.bz2",
        "speed": 0.94,
    },
    "Arthur (Authoritative UK Male - Nature)": {
        "folder": "vits-piper-en_GB-alan-medium",
        "model": "en_GB-alan-medium.onnx",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_GB-alan-medium.tar.bz2",
        "speed": 0.95,
    },
    "Elena (Warm US Female - Storytelling)": {
        "folder": "vits-piper-en_US-amy-medium",
        "model": "en_US-amy-medium.onnx",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-medium.tar.bz2",
        "speed": 0.96,
    },
    "Clara (Crisp UK Female - Journalism)": {
        "folder": "vits-piper-en_GB-jenny_dioco-medium",
        "model": "en_GB-jenny_dioco-medium.onnx",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_GB-jenny_dioco-medium.tar.bz2",
        "speed": 0.97,
    },
    "Julian (Smooth US Male - True Crime)": {
        "folder": "vits-piper-en_US-joe-medium",
        "model": "en_US-joe-medium.onnx",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-joe-medium.tar.bz2",
        "speed": 0.92,
    },
}


class DocuVoiceApp(App):
    def build(self):
        self.title = "DocuVoice Studio (Offline)"
        self.tts_engine = None
        self.loaded_voice_key = None
        self.current_audio_path = None
        self.sound_obj = None

        root = BoxLayout(orientation="vertical", padding=24, spacing=16)

        header = Label(
            text="[b]DocuVoice AI Studio[/b]\n5 Pre-Bundled Offline Documentary Voices",
            markup=True,
            font_size="20sp",
            size_hint_y=None,
            height=90,
            halign="center",
        )
        root.add_widget(header)

        self.script_input = TextInput(
            hint_text="Paste your documentary script here...",
            multiline=True,
            font_size="16sp",
            size_hint_y=0.45,
        )
        root.add_widget(self.script_input)

        row1 = BoxLayout(orientation="horizontal", spacing=12, size_hint_y=None, height=85)
        self.voice_spinner = Spinner(
            text="Marcus (Deep US Male - History)",
            values=list(VOICES.keys()),
            font_size="14sp",
            size_hint_x=0.68,
        )
        self.voice_spinner.bind(text=self.on_voice_change)

        self.format_spinner = Spinner(
            text="MP3",
            values=["WAV", "MP3", "OGG", "FLAC"],
            font_size="14sp",
            size_hint_x=0.32,
        )
        row1.add_widget(self.voice_spinner)
        row1.add_widget(self.format_spinner)
        root.add_widget(row1)

        speed_box = BoxLayout(orientation="vertical", size_hint_y=None, height=85)
        self.speed_label = Label(text="Narration Pacing: 0.94x", font_size="14sp")
        self.speed_slider = Slider(min=0.75, max=1.25, value=0.94, step=0.01)
        self.speed_slider.bind(value=self.on_speed_change)
        speed_box.add_widget(self.speed_label)
        speed_box.add_widget(self.speed_slider)
        root.add_widget(speed_box)

        self.status_label = Label(
            text="Ready — All 5 voices bundled for 100% offline use.",
            font_size="13sp",
            size_hint_y=None,
            height=60,
        )
        self.progress = ProgressBar(max=100, value=0, size_hint_y=None, height=20)
        root.add_widget(self.status_label)
        root.add_widget(self.progress)

        btn_row = BoxLayout(orientation="horizontal", spacing=12, size_hint_y=None, height=95)
        self.gen_btn = Button(
            text="Generate & Export",
            font_size="16sp",
            background_color=(0.15, 0.55, 0.95, 1),
        )
        self.gen_btn.bind(on_press=self.start_generation)

        self.play_btn = Button(
            text="Play Preview",
            font_size="16sp",
            disabled=True,
            background_color=(0.2, 0.75, 0.45, 1),
        )
        self.play_btn.bind(on_press=self.toggle_play)

        btn_row.add_widget(self.gen_btn)
        btn_row.add_widget(self.play_btn)
        root.add_widget(btn_row)

        return root

    def on_voice_change(self, spinner, text):
        self.speed_slider.value = VOICES[text]["speed"]

    def on_speed_change(self, slider, val):
        self.speed_label.text = f"Narration Pacing: {val:.2f}x"

    def set_status(self, msg, progress_val=None):
        def _update(dt):
            self.status_label.text = msg
            if progress_val is not None:
                self.progress.value = progress_val
        Clock.schedule_once(_update)

    def resolve_voice_dir(self, voice_key):
        cfg = VOICES[voice_key]
        folder = cfg["folder"]

        bundled_candidates = [
            os.path.join(self.directory, "assets", "models", folder),
            os.path.join(os.getcwd(), "assets", "models", folder),
        ]
        for candidate in bundled_candidates:
            if os.path.exists(os.path.join(candidate, cfg["model"])):
                return candidate

        models_dir = os.path.join(self.user_data_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        voice_dir = os.path.join(models_dir, folder)
        model_path = os.path.join(voice_dir, cfg["model"])

        if not os.path.exists(model_path):
            self.set_status(f"Downloading {voice_key.split()[0]} voice...", 15)
            tar_path = os.path.join(models_dir, f"{folder}.tar.bz2")
            urllib.request.urlretrieve(cfg["url"], tar_path)
            self.set_status("Extracting voice pack...", 35)
            with tarfile.open(tar_path, "r:bz2") as tar:
                tar.extractall(path=models_dir)
            os.remove(tar_path)

        return voice_dir

    def ensure_voice_loaded(self, voice_key):
        cfg = VOICES[voice_key]
        voice_dir = self.resolve_voice_dir(voice_key)
        model_path = os.path.join(voice_dir, cfg["model"])

        if self.loaded_voice_key != voice_key or self.tts_engine is None:
            self.set_status("Loading neural voice engine...", 45)
            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=model_path,
                        tokens=os.path.join(voice_dir, "tokens.txt"),
                        data_dir=os.path.join(voice_dir, "espeak-ng-data"),
                    ),
                    num_threads=4,
                    provider="cpu",
                )
            )
            self.tts_engine = sherpa_onnx.OfflineTts(tts_config)
            self.loaded_voice_key = voice_key

    def start_generation(self, instance):
        script = self.script_input.text.strip()
        if not script:
            self.set_status("Please enter a documentary script first.")
            return

        self.gen_btn.disabled = True
        self.play_btn.disabled = True
        threading.Thread(target=self._run_tts_pipeline, args=(script,), daemon=True).start()

    def _run_tts_pipeline(self, script):
        try:
            voice_key = self.voice_spinner.text
            fmt = self.format_spinner.text.lower()
            speed = float(self.speed_slider.value)

            self.ensure_voice_loaded(voice_key)

            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", script) if s.strip()]
            segments = []
            sample_rate = 22050
            silence = np.zeros(int(sample_rate * 0.35), dtype=np.float32)

            for idx, sentence in enumerate(sentences):
                pct = 55 + int((idx / max(len(sentences), 1)) * 35)
                self.set_status(f"Synthesizing sentence {idx + 1}/{len(sentences)}...", pct)
                audio = self.tts_engine.generate(sentence, sid=0, speed=speed)
                sample_rate = audio.sample_rate
                segments.append(np.array(audio.samples, dtype=np.float32))
                if idx < len(sentences) - 1:
                    segments.append(silence)

            full_audio = np.concatenate(segments)

            out_dir = "/storage/emulated/0/Download/DocuVoice"
            os.makedirs(out_dir, exist_ok=True)
            out_path = export_audio_file(full_audio, sample_rate, out_dir, fmt)

            self.current_audio_path = out_path
            self.set_status(f"Saved to: Download/DocuVoice/{os.path.basename(out_path)}", 100)
            Clock.schedule_once(lambda dt: setattr(self.play_btn, "disabled", False))

        except Exception as e:
            self.set_status(f"Error: {str(e)}", 0)
        finally:
            Clock.schedule_once(lambda dt: setattr(self.gen_btn, "disabled", False))

    def toggle_play(self, instance):
        if not self.current_audio_path or not os.path.exists(self.current_audio_path):
            return
        if self.sound_obj and self.sound_obj.state == "play":
            self.sound_obj.stop()
            self.play_btn.text = "Play Preview"
        else:
            self.sound_obj = SoundLoader.load(self.current_audio_path)
            if self.sound_obj:
                self.sound_obj.play()
                self.play_btn.text = "Stop Preview"


if __name__ == "__main__":
    DocuVoiceApp().run()
