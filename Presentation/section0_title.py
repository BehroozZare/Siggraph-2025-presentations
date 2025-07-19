from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService


# 2) extend the pre-amble
template = TexTemplate()
template.add_to_preamble(r"\usepackage{xcolor}")

config.tex_template = template


# Section1: Handles the animation and logic for the problem definition section of the FastTrack video.
class Section0():
    def __init__(self, scene: Scene | VoiceoverScene):
        self.scene = scene
        self.write_time = 0.5
        self.wait_time = 2
        self.text_color = BLACK

    def make_paper_title(self) -> Tex:
        # Create the title for the section
        title = Tex(r"Adaptive Algebraic Reuse of Reordering in Cholesky Factorizations \\ with Dynamic Sparsity Patterns", color=self.text_color, font_size=36)
        return title
    
    def make_author_names(self) -> Tex:
        author_names = Tex(r"B. Zarebavani, D.M. Kaufman, D.I.W. Levin, M.M. Dehnavi", color=self.text_color, font_size=24)
        return author_names

    
    def _play_scene(self):
      title = self.make_paper_title()
      title.center()
      author_names = self.make_author_names()
      author_names.next_to(title, DOWN)
      self.scene.play(Write(title), Write(author_names), run_time=self.write_time)
      self.scene.wait(self.wait_time)

    def play_scene(self):
        # Entry point for Section 1 animation, with or without voiceover
        if isinstance(self.scene, VoiceoverScene):
            scripts = "This is the 'adaptive algebraic reuse of reordering in Cholesky factorizations with dynamic-sparsity-patterns'."
            with self.scene.voiceover(text=scripts) as tracker:    
                self._play_scene()
        else:
            self._play_scene()