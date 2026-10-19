"""
Dịch vụ Chuyển văn bản thành giọng nói (Text-to-Speech - TTS) Thuyết minh Di sản Champa:
1. Engine 1 (Chính): Piper TTS (vi_VN-vais1000-medium) - Mã nguồn mở, MIT, siêu tốc trên CPU.
2. Engine 2 (Đối chứng): MMS-TTS (facebook/mms-tts-vie) - Meta AI VITS qua Transformers, CC-BY-NC.
3. Tiền xử lý chuẩn hóa văn bản nói (Text Normalizer): Lọc sạch thẻ trích dẫn, khử định dạng markdown,
   phiên âm niên đại thế kỷ La Mã thành số tiếng Việt tự nhiên.
4. Cơ chế Bộ đệm (Audio Cache): Lưu trữ file WAV giúp phát lại tức thì (<1ms).
"""
import io
import os
import re
import sys
import wave
import time
import hashlib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch

class ChampaTTSService:
    def __init__(
        self,
        piper_model_path: str = "models/piper_voices/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx",
        piper_config_path: str = "models/piper_voices/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json",
        cache_dir: str = "data/audio_cache"
    ):
        self.piper_model_path = Path(piper_model_path)
        self.piper_config_path = Path(piper_config_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Lazy loading
        self._piper_voice = None
        self._mms_model = None
        self._mms_tokenizer = None

    def clean_text_for_speech(self, text: str) -> str:
        """
        Làm sạch và chuẩn hóa văn bản trước khi đưa vào bộ tổng hợp giọng nói:
        - Xóa triệt để các thẻ trích dẫn [DOC...] và ghi chú học thuật
        - Khử markdown headers, bold, bullet points
        - Chuẩn hóa thế kỷ La Mã thành cách đọc tự nhiên trong tiếng Việt
        """
        # 1. Loại bỏ các thẻ trích dẫn tài liệu như [DOC01_TREASURES_C04]
        t = re.sub(r'\[DOC\w+\]', '', text)
        
        # 2. Loại bỏ các phần chú thích kỹ thuật, verifier, metadata tags
        t = re.sub(r'\*Lưu ý kiểm định:.*?\*', '', t)
        t = re.sub(r'#{1,6}\s*', '', t)
        t = re.sub(r'\*\*([^*]+)\*\*', r'\1', t)
        t = re.sub(r'\*([^*]+)\*', r'\1', t)
        t = re.sub(r'`([^`]+)`', r'\1', t)
        
        # 3. Chuẩn hóa tiêu đề phân mục La Mã
        t = re.sub(r'\bI\.\s*', 'Phần một: ', t)
        t = re.sub(r'\bII\.\s*', 'Phần hai: ', t)
        t = re.sub(r'\bIII\.\s*', 'Phần ba: ', t)
        t = re.sub(r'\bIV\.\s*', 'Phần bốn: ', t)
        
        # 4. Phiên âm khoảng thế kỷ La Mã sang số để TTS đọc tự nhiên
        roman_ranges = [
            (r'\bThế kỷ VII\s*–\s*VIII\b', 'Thế kỷ bảy đến thế kỷ tám'),
            (r'\bThế kỷ VII\s*-\s*VIII\b', 'Thế kỷ bảy đến thế kỷ tám'),
            (r'\bThế kỷ VIII\s*–\s*IX\b', 'Thế kỷ tám đến thế kỷ chín'),
            (r'\bThế kỷ VIII\s*-\s*IX\b', 'Thế kỷ tám đến thế kỷ chín'),
            (r'\bThế kỷ IX\s*–\s*X\b', 'Thế kỷ chín đến thế kỷ mười'),
            (r'\bThế kỷ IX\s*-\s*X\b', 'Thế kỷ chín đến thế kỷ mười'),
            (r'\bThế kỷ X\s*–\s*XI\b', 'Thế kỷ mười đến thế kỷ mười một'),
            (r'\bThế kỷ X\s*-\s*XI\b', 'Thế kỷ mười đến thế kỷ mười một'),
            (r'\bThế kỷ XI\s*–\s*XII\b', 'Thế kỷ mười một đến thế kỷ mười hai'),
            (r'\bThế kỷ XI\s*-\s*XII\b', 'Thế kỷ mười một đến thế kỷ mười hai'),
            (r'\bThế kỷ XII\s*–\s*XIII\b', 'Thế kỷ mười hai đến thế kỷ mười ba'),
            (r'\bThế kỷ XII\s*-\s*XIII\b', 'Thế kỷ mười hai đến thế kỷ mười ba'),
            (r'\bThế kỷ VII\b', 'Thế kỷ bảy'),
            (r'\bThế kỷ VIII\b', 'Thế kỷ tám'),
            (r'\bThế kỷ IX\b', 'Thế kỷ chín'),
            (r'\bThế kỷ X\b', 'Thế kỷ mười'),
            (r'\bThế kỷ XI\b', 'Thế kỷ mười một'),
            (r'\bThế kỷ XII\b', 'Thế kỷ mười hai'),
            (r'\bThế kỷ XIII\b', 'Thế kỷ mười ba'),
        ]
        for pattern, replacement in roman_ranges:
            t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
            
        # 5. Dấu gạch đầu dòng và ngắt dòng
        t = re.sub(r'^\s*[-*•]\s*', '', t, flags=re.MULTILINE)
        t = re.sub(r'\n+', '. ', t)
        t = re.sub(r'\s*([.,;:!?])\s*', r'\1 ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        
        return t

    def _get_cache_path(self, text: str, engine: str) -> Path:
        """Tạo đường dẫn bộ đệm dựa trên băm MD5 của văn bản và tên engine"""
        h = hashlib.md5(f"{engine}_{text}".encode('utf-8')).hexdigest()
        return self.cache_dir / f"{engine}_{h}.wav"

    def _load_piper(self):
        if self._piper_voice is None:
            from piper import PiperVoice
            if not self.piper_model_path.exists():
                raise FileNotFoundError(f"Không tìm thấy mô hình Piper tại: {self.piper_model_path}")
            self._piper_voice = PiperVoice.load(
                str(self.piper_model_path),
                config_path=str(self.piper_config_path) if self.piper_config_path.exists() else None
            )
        return self._piper_voice

    def _load_mms(self):
        if self._mms_model is None or self._mms_tokenizer is None:
            from transformers import VitsModel, AutoTokenizer
            self._mms_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
            self._mms_model = VitsModel.from_pretrained("facebook/mms-tts-vie").eval()
        return self._mms_model, self._mms_tokenizer

    def synthesize_piper(self, text: str) -> Tuple[bytes, float, int]:
        """Tổng hợp âm thanh bằng Piper TTS (Mô hình chính, siêu tốc trên CPU)"""
        voice = self._load_piper()
        wav_io = io.BytesIO()
        t0 = time.time()
        with wave.open(wav_io, 'wb') as wav_file:
            voice.synthesize_wav(text, wav_file)
        latency = time.time() - t0
        wav_bytes = wav_io.getvalue()
        sr = voice.config.sample_rate
        return wav_bytes, latency, sr

    def synthesize_mms(self, text: str) -> Tuple[bytes, float, int]:
        """Tổng hợp âm thanh bằng Meta MMS-TTS (Mô hình đối chứng học thuật)"""
        model, tokenizer = self._load_mms()
        t0 = time.time()
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform
        latency = time.time() - t0
        
        # Chuyển tensor sang WAV bytes
        audio_arr = output.squeeze().cpu().numpy()
        audio_int16 = (audio_arr * 32767).clip(-32768, 32767).astype(np.int16)
        
        sr = model.config.sampling_rate
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sr)
            wav_file.writeframes(audio_int16.tobytes())
            
        return wav_io.getvalue(), latency, sr

    def synthesize(self, text: str, engine: str = "piper", use_cache: bool = True) -> Dict[str, Any]:
        """
        Hàm giao tiếp chính: tổng hợp văn bản thành giọng nói có hỗ trợ bộ đệm.
        engine: 'piper' hoặc 'mms'
        """
        cleaned_text = self.clean_text_for_speech(text)
        if not cleaned_text:
            return {"error": "Văn bản rỗng sau khi làm sạch."}

        cache_path = self._get_cache_path(cleaned_text, engine)
        if use_cache and cache_path.exists():
            wav_bytes = cache_path.read_bytes()
            # Đọc sample rate và thời lượng
            with wave.open(io.BytesIO(wav_bytes), 'rb') as w:
                sr = w.getframerate()
                n_frames = w.getnframes()
                duration = n_frames / float(sr)
            return {
                "wav_bytes": wav_bytes,
                "latency_sec": 0.001,
                "duration_sec": duration,
                "sample_rate": sr,
                "cached": True,
                "engine": engine,
                "cleaned_text": cleaned_text
            }

        if engine == "mms":
            wav_bytes, latency, sr = self.synthesize_mms(cleaned_text)
        else: # default piper
            wav_bytes, latency, sr = self.synthesize_piper(cleaned_text)

        # Lưu bộ đệm
        try:
            cache_path.write_bytes(wav_bytes)
        except Exception:
            pass

        with wave.open(io.BytesIO(wav_bytes), 'rb') as w:
            n_frames = w.getnframes()
            duration = n_frames / float(sr)

        rtf = latency / max(duration, 0.001)

        return {
            "wav_bytes": wav_bytes,
            "latency_sec": latency,
            "duration_sec": duration,
            "sample_rate": sr,
            "rtf": rtf,
            "cached": False,
            "engine": engine,
            "cleaned_text": cleaned_text
        }

