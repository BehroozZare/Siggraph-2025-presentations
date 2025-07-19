
from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService

# 2) extend the pre-amble
template = TexTemplate()
template.add_to_preamble(r"\usepackage{xcolor}")

config.tex_template = template

class Section3():
    def __init__(self, scene: Scene | VoiceoverScene):
        self.scene = scene
        self.FadeIn_time = 0.5
        self.section3_total_time = 2
        
        
    def _parth_paper_title(self):
        title = Tex(r"Adaptive Algebraic Reuse of Reordering in Cholesky Factorizations with Dynamic Sparsity Patterns", font_size=36)
        return title
    
    def _parth_site(self):
        subtitle = Tex(r"github.com/behrooz/parth",  font_size=36, color=BLUE)
        return subtitle
    
    def _run_scene(self):
        paper_title = self._parth_paper_title()
        paper_title.to_edge(UP, buff=1)
        # QR code link
        qr_code = ImageMobject("Figures/section3/qr_code.png")
        qr_code.scale_to_fit_width(paper_title.get_width() * 0.2)
        qr_code.next_to(paper_title, DOWN, buff=1)

    
        # Show elements
        self.scene.play(FadeIn(paper_title), FadeIn(qr_code), runtime=self.FadeIn_time)
        self.scene.wait(self.section3_total_time - self.FadeIn_time)
    
    def section3(self):
        # Main title
        if isinstance(self.scene, VoiceoverScene):
            with self.scene.voiceover(text="Thank you and see you in our session!") as tracker:
                self._run_scene()
        else:
            self._run_scene()
