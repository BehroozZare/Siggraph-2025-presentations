from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
from section0_title import Section0
from section1_background import Section1


class FastTrackComplete(VoiceoverScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.section0 = Section0(self)
        self.section1 = Section1(self)
        # self.section2 = Section2(self)
        # self.section3 = Section3(self)    

    def construct(self):
        # set your TTS engine once
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE
        # self.set_speech_service(RecorderService())
        # — Section 1 —
        # self.section0.play_scene()
        # if self.mobjects:
        #     self.play(FadeOut(*self.mobjects), run_time=0.2)
        #     self.wait(0.1)

        self.section1.play_scene()
        # Clear the scene
        if self.mobjects:
            self.play(FadeOut(*self.mobjects), run_time=0.2)
            self.wait(0.1)

        # # # — Section 2 —
        # self.section2.section2()
        # if self.mobjects:
        #     self.play(FadeOut(*self.mobjects), run_time=0.2)
        #     self.wait(0.1)

        # #— Section 3 —
        # self.section3.section3()
        # if self.mobjects:
        #     self.wait(0.1)