from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
from section0_title import Title
from section1_background import Background
from section2_problem_definition import ProblemDefinition
from section3_parth_solution import ParthSolution
from section4_results import Results
from section5_ending import ThankYouNote


class FastTrackComplete(VoiceoverScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.section0 = Title(self)
        self.section1 = Background(self)
        self.section2 = ProblemDefinition(self)
        self.section3 = ParthSolution(self)
        self.section4 = Results(self)
        self.section5 = ThankYouNote(self)
        # self.section2 = Section2(self)
        # self.section3 = Section3(self)    

    def construct(self):
        # set your TTS engine once
        self.set_speech_service(GTTSService())
        self.camera.background_color = WHITE
        # self.set_speech_service(RecorderService())
        # — Section 1 —
        # self.section0.play_title()
        # if self.mobjects:
        #     self.play(FadeOut(*self.mobjects), run_time=0.2)
        #     self.wait(0.1)

        # self.section1.play_background()
        # # Clear the scene
        # if self.mobjects:
        #     self.play(FadeOut(*self.mobjects), run_time=0.2)
        #     self.wait(0.1)

        # — Section 2 —
        self.section2.play_problem_definition()
        if self.mobjects:
            self.play(FadeOut(*self.mobjects), run_time=0.2)
            self.wait(0.1)

        # # #— Section 3 —
        # self.section3.play_parth_solution()
        # if self.mobjects:
        #     self.play(FadeOut(*self.mobjects), run_time=0.2)
        #     self.wait(0.1)

        # # — Section 4 —
        # self.section4.play_results()
        # if self.mobjects:
        #     self.play(FadeOut(*self.mobjects), run_time=0.2)
        #     self.wait(0.1)

        # # — Section 5 —
        # self.section5.play_thank_you_note()
        # if self.mobjects:
        #     self.play(FadeOut(*self.mobjects), run_time=0.2)
        #     self.wait(0.1)