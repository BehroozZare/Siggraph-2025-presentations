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

from section1_background import Background



class Scene_1_1_CowExample(VoiceoverScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background = Background(self)

    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(SERVICE)
        self.background.play_cow_scene(self)


class Scene_1_2_FrameworkIntroduction(VoiceoverScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background = Background(self)
    
    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(SERVICE)
        self.background.play_framework_introduction_scene(self)


class Scene_1_3_ExampleOfExpensiveSymbolicAnalysis(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background = Background(self)
    
    def construct(self):
        self.camera.background_color = WHITE
        self.background.play_example_of_expensive_symbolic_analysis(self)


class Scene_1_4_ComputationalFlow(VoiceoverScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background = Background(self)
    
    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(SERVICE)
        self.background.play_computational_flow_scene(self)






