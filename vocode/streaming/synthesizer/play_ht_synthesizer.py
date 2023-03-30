import io
from typing import Optional
from pydub import AudioSegment

import requests
from vocode import getenv
from vocode.streaming.agent.bot_sentiment_analyser import BotSentiment
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import PlayHtSynthesizerConfig
from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    SynthesisResult,
    encode_as_wav,
)
from vocode.streaming.utils import convert_wav

TTS_ENDPOINT = "https://play.ht/api/v2/tts/stream"


class PlayHtSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        synthesizer_config: PlayHtSynthesizerConfig,
        api_key: str = None,
        user_id: str = None,
    ):
        super().__init__(synthesizer_config)
        self.synthesizer_config = synthesizer_config
        self.api_key = api_key or getenv("PLAY_HT_API_KEY")
        self.user_id = user_id or getenv("PLAY_HT_USER_ID")
        if not self.api_key or not self.user_id:
            raise ValueError(
                "You must set the PLAY_HT_API_KEY and PLAY_HT_USER_ID environment variables"
            )

    def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        bot_sentiment: Optional[BotSentiment] = None,
    ) -> SynthesisResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-User-ID": self.user_id,
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
        }
        body = {
            "voice": self.synthesizer_config.voice_id,
            "text": message.text,
            "sample_rate": self.synthesizer_config.sampling_rate,
        }
        if self.synthesizer_config.speed:
            body["speed"] = self.synthesizer_config.speed
        if self.synthesizer_config.preset:
            body["preset"] = self.synthesizer_config.preset

        response = requests.post(TTS_ENDPOINT, headers=headers, json=body, timeout=5)
        if not response.ok:
            print(response.text)

        audio_segment: AudioSegment = AudioSegment.from_mp3(
            io.BytesIO(response.content)
        )

        output_bytes_io = io.BytesIO()
        audio_segment.export(output_bytes_io, format="wav")

        if self.synthesizer_config.audio_encoding == AudioEncoding.LINEAR16:
            output_bytes = convert_wav(
                output_bytes_io,
                output_sample_rate=self.synthesizer_config.sampling_rate,
                output_encoding=AudioEncoding.LINEAR16,
            )
        elif self.synthesizer_config.audio_encoding == AudioEncoding.MULAW:
            output_bytes = convert_wav(
                output_bytes_io,
                output_sample_rate=self.synthesizer_config.sampling_rate,
                output_encoding=AudioEncoding.MULAW,
            )

        if self.synthesizer_config.should_encode_as_wav:
            output_bytes = encode_as_wav(output_bytes)

        def chunk_generator(output_bytes):
            for i in range(0, len(output_bytes), chunk_size):
                if i + chunk_size > len(output_bytes):
                    yield SynthesisResult.ChunkResult(output_bytes[i:], True)
                else:
                    yield SynthesisResult.ChunkResult(
                        output_bytes[i : i + chunk_size], False
                    )

        return SynthesisResult(
            chunk_generator(output_bytes),
            lambda seconds: self.get_message_cutoff_from_total_response_length(
                message, seconds, len(output_bytes)
            ),
        )
