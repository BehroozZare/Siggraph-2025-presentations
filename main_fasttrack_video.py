from manim import *
from section1_problem_definition import ProblemDefinition
from section2_parth_solution import ParthContribution
from section3_finish import Outro

class FastTrackComplete(Scene):
    """
    Complete FastTrack video combining all 5 sections
    Target duration: ~20 seconds
    """
    
    def construct(self):
        # Section 1: Problem Definition (3-4 seconds)
        self.run_section(ProblemDefinition)
        
        # Section 2: Parth's Solution (3-4 seconds)
        self.run_section(ParthContribution)

        # Section 3: Ending
        self.run_section(Outro)


        
    
    def run_section(self, section_class):
        """Helper to run a section class within this scene"""
        # Clear any existing mobjects
        if self.mobjects:
            self.play(FadeOut(*self.mobjects), run_time=0.3)
            self.wait(0.1)
        
        # Create temporary scene instance
        temp_scene = section_class()
        
        # Copy essential scene methods and properties to temp scene
        # This allows the section to use this scene's context
        temp_scene.mobjects = self.mobjects
        temp_scene.add = self.add
        temp_scene.remove = self.remove
        temp_scene.play = self.play
        temp_scene.wait = self.wait
        temp_scene.clear = self.clear
        
        # Run the section's construct method
        temp_scene.construct() 