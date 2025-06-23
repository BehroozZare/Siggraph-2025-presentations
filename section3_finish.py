from manim import *

class Outro(Scene):

    def _parth_paper_title(self):
        title = Tex(r"Adaptive Algebraic Reuse of Reordering in Cholesky Factorizations with Dynamic Sparsity Patterns", font_size=36)
        return title
    
    def _parth_site(self):
        subtitle = Tex(r"github.com/behrooz/parth",  font_size=36, color=BLUE)
        return subtitle
    
    def construct(self):
        # Main title
        paper_title = self._parth_paper_title()
        paper_title.to_edge(UP, buff=1)
        
        # Subtitle
        paper_site = self._parth_site()
        paper_site.next_to(paper_title, DOWN, buff=0.5)
        
        
        # Show elements
        self.play(Write(paper_title))
        self.wait(0.5)
        self.play(Write(paper_site))
        self.wait(0.5)