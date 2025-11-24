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

from section2_problem_definition import ProblemDefinition


class Scene_2_1_DynamicSparsityPatternExample(VoiceoverScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background = ProblemDefinition(self)

    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(SERVICE)
        self.background.play_dynamic_sparsity_pattern_example(self)


class Scene_2_2_SymbolicNumericDynamicSparsityPatternFlow(VoiceoverScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background = ProblemDefinition(self)

    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(SERVICE)
        self.background.play_symbolic_numeric_dynamic_sparsity_pattern_flow(self)


class Scene_2_3_SymbolicAnalysisOverhead(VoiceoverScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background = ProblemDefinition(self)

    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(SERVICE)
        self.background.play_show_symbolic_analysis_chart(self)

class Scene_2_4_InternalComponents(VoiceoverScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background = ProblemDefinition(self)

    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(SERVICE)
        self.background.play_internal_components(self)


class Scene_2_5_InternalBenchmark(VoiceoverScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background = ProblemDefinition(self)

    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(SERVICE)
        self.background.play_show_internal_overhead_chart(self)

class Scene_2_6_TheObjectiveOfOrdering(VoiceoverScene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background = ProblemDefinition(self)

    def construct(self):
        self.camera.background_color = WHITE
        self.set_speech_service(SERVICE)
        self.background.play_the_objective_of_ordering(self)
