from manim import *
from section1_background import Section1
from section2_problem_definition import Section2
from Presentation.src.section5_ending import Section3

class FastTrackComplete(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.section1 = Section1(self)
        self.section2 = Section2(self)
        self.section3 = Section3(self)    

    def construct(self):
        # — Section 1 —
        self.section1.section1()
        # Clear the scene
        if self.mobjects:
            self.play(FadeOut(*self.mobjects), run_time=0.2)
            self.wait(0.1)

        # # — Section 2 —
        self.section2.section2()
        if self.mobjects:
            self.play(FadeOut(*self.mobjects), run_time=0.2)
            self.wait(0.1)

        #— Section 3 —
        self.section3.section3()
        if self.mobjects:
            self.wait(0.1)