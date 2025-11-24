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

from section3_parth_solution import ParthSolution



class Scene_3_1_ParthIntroduction(VoiceoverScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solution = ParthSolution(self)

    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(SERVICE)
        self.solution.parth_introduction()


class Scene_3_2_ModulesIntroduction(VoiceoverScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solution = ParthSolution(self)

    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(SERVICE)
        self.solution.modules_introduction()


class Scene_3_3_FirstCallToHGD(VoiceoverScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solution = ParthSolution(self)

    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(SERVICE)
        self.solution.first_call_to_hgd()


class Scene_3_4_FirstCallToAssembler(VoiceoverScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solution = ParthSolution(self)

    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(SERVICE)
        self.solution.first_call_to_assembler()

class Scene_3_5_ChangeIntegration(VoiceoverScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.solution = ParthSolution(self)

    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(SERVICE)
        self.solution.change_integration()

