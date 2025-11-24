from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService

import sys
import os

SERVICE = RecorderService()
# SERIVCE = GTTSService()

# Add the src directory to Python path when running from Presentation folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from section4_results import Results





class Scene_4_1_Results(VoiceoverScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.results = Results(self)

    def construct(self):
        self.set_speech_service(SERVICE)
        self.camera.background_color = WHITE
        self.results.play_results()


