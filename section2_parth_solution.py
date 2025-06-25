from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService

# 2) extend the pre-amble
template = TexTemplate()
template.add_to_preamble(r"\usepackage{listings}")
template.add_to_preamble(r"\usepackage{xcolor}")
template.add_to_preamble(r"\lstset { language=C++, basicstyle=\footnotesize}")

config.tex_template = template


# Configure LaTeX template to include xcolor package
config.tex_template.add_to_preamble(r"\usepackage{xcolor}")

class ParthContribution(VoiceoverScene):

    def _parth_3_lines_integration(self) -> Code:
        # Show the three lines of code
        # Parth parth;
        # parth.setMatrix(matrix);
        # parth.computePermutation();
        # with latex code block in manim

        code_block ="""
            Parth parth;
            parth.setMatrix(matrix);
            parth.computePermutation();"""

        
        rendered_cpp_code = Code(
            code_string=code_block,
            language="cpp",
            background="window",
            paragraph_config={"line_spacing": 1, "font": "Noto Sans Mono", "font_size": 24}
        )
        self.add(rendered_cpp_code)

        return rendered_cpp_code


    def _parth_intergration_block(self, font_size=20, inner_rect_width=4, inner_rect_height=1.2, inner_rect_left_right_offset=3, inner_rect_top_offset=1.5) -> VGroup:
        # create a big transparent rectangle with a border
        # within that rectangle, there is a figure  (Figures/section2/brain.svg) on top with three arrow comming out of it to 3 other
        # rectangles withing that block. The name of each rectangle is Accelerate, CHOLMOD, and MKL Pardiso

        # Create the main container rectangle
        container = Rectangle(
            width=10, 
            height=6, 
            stroke_color=WHITE, 
            stroke_width=2, 
            fill_opacity=0.2,
            fill_color=WHITE
        )
        
        # Load the brain SVG and position it on the middle left side
        brain = SVGMobject("Figures/section2/brain.svg")
        brain.set_height(1.5)
        brain.move_to(container.get_left() + RIGHT * 2)
        #Add a subtitle to the brain
        brain_subtitle = Text("Parth", font_size=font_size, color=WHITE)
        brain_subtitle.move_to(brain.get_bottom() + DOWN * 0.5)
        
        # Create three rectangles for the libraries in a column on the right side
        accelerate_rect = Rectangle(
            width=inner_rect_width, 
            height=inner_rect_height, 
            stroke_color=WHITE, 
            stroke_width=1.5,
            fill_opacity=0.3,
            fill_color=GREEN
        )
        accelerate_rect.move_to(container.get_right() + LEFT * inner_rect_left_right_offset + UP * inner_rect_top_offset)
        
        cholmod_rect = Rectangle(
            width=inner_rect_width, 
            height=inner_rect_height, 
            stroke_color=WHITE, 
            stroke_width=1.5,
            fill_opacity=0.3,
            fill_color=YELLOW
        )
        cholmod_rect.move_to(container.get_right() + LEFT * inner_rect_left_right_offset)
        
        mkl_rect = Rectangle(
            width=inner_rect_width, 
            height=inner_rect_height, 
            stroke_color=WHITE, 
            stroke_width=1.5,
            fill_opacity=0.4,
            fill_color=RED
        )
        mkl_rect.move_to(container.get_right() + LEFT * inner_rect_left_right_offset + DOWN * inner_rect_top_offset)
        
        # Add text labels to the rectangles
        accelerate_text = Text("Accelerate", font_size=font_size, color=WHITE)
        accelerate_text.move_to(accelerate_rect.get_center())
        
        cholmod_text = Text("CHOLMOD", font_size=font_size, color=WHITE)
        cholmod_text.move_to(cholmod_rect.get_center())
        
        mkl_text = Text("MKL Pardiso", font_size=font_size, color=WHITE)
        mkl_text.move_to(mkl_rect.get_center())
        
        # Create arrows from brain to each rectangle (horizontal arrows)
        arrow1 = Arrow(
            brain.get_right() + RIGHT * 0.2,
            accelerate_rect.get_left() + LEFT * 0.1,
            buff=0.1,
            stroke_width=2,
            color=WHITE
        )
        
        arrow2 = Arrow(
            brain.get_right() + RIGHT * 0.2,
            cholmod_rect.get_left() + LEFT * 0.1,
            buff=0.1,
            stroke_width=2,
            color=WHITE
        )
        
        arrow3 = Arrow(
            brain.get_right() + RIGHT * 0.2,
            mkl_rect.get_left() + LEFT * 0.1,
            buff=0.1,
            stroke_width=2,
            color=WHITE
        )
        
        # Group all elements together
        integration_block = VGroup(
            container,
            brain,
            brain_subtitle,
            accelerate_rect,
            cholmod_rect,
            mkl_rect,
            accelerate_text,
            cholmod_text,
            mkl_text,
            arrow1,
            arrow2,
            arrow3
        )
        
        return integration_block

    def _parth_reuse_example(self):
        # Add the figure in Figures/section2/FastTrack.png
        fast_track = ImageMobject("Figures/section2/FastTrack.png")
        # fit the bounding box of the image to the image

        return fast_track

    def _parth_site_link(self):
        pass

    def construct(self):
        self.set_speech_service(GTTSService())

        with self.voiceover(text="Parth addresses this issue by only three lines of additional code!") as tracker:
            code_block = self._parth_3_lines_integration()
            code_block.to_edge(UP)
            code_block.to_edge(LEFT)
            self.play(FadeIn(code_block))


        #Arrow from code_block to the next block
        integration_block = self._parth_intergration_block(font_size=32)
        # Make the width of the integration_block the same as the code_block
        integration_block.scale_to_fit_height(code_block.get_height())
        integration_block.to_edge(UP)
        integration_block.to_edge(RIGHT)

        with self.voiceover(text="By integrating Parth into well-known Cholesky solvers, we enhanced them with a smart memory!") as tracker:
                #Create an arrow between the code_block and the integration_block
                arrow = Arrow(code_block.get_right(), integration_block.get_left(), color=WHITE)
                self.play(Create(arrow))

                self.add(integration_block)
                self.play(Create(integration_block), run_time=2)



        reuse_example = self._parth_reuse_example()
        reuse_example.scale_to_fit_width(integration_block.get_width())
        reuse_example.next_to(integration_block, DOWN, buff=1.5)
        

        with self.voiceover(text="Which allows for reuse of computation!") as tracker:
            #Arrow from the integration_block to the reuse_example
            arrow = Arrow(integration_block.get_bottom(), reuse_example.get_top(), color=WHITE)
            self.play(Create(arrow))

            #Animate the reuse_example photo
            self.play(FadeIn(reuse_example))



        init_values = [0, 0, 0]
        final_values = [5.9, 3.8, 2.8]
        # Create the bar chart that shows the inspector bottleneck
        chart = BarChart(
            values=init_values,
            bar_names=["Accelerate", "CHOLMOD", "MKL Pardiso"],
            y_range=[0, 6, 1],
            y_length=2,
            x_length=3,
            x_axis_config={
                "font_size": 22,
                "label_constructor": lambda text: Tex(text, font_size=22)  # Move labels down by 0.3 units
            },
            y_axis_config={"font_size": 22}
        )
        # Get the x position of the code block and the y position of the reuse_example
        chart.scale_to_fit_height(reuse_example.get_height())
        chart.next_to(code_block, DOWN, buff=1.5)
        
        
        # Manually rotate the x-axis labels by 45 degrees
        for label in chart.x_axis.labels:
            label.rotate(PI/4)  # 45 degrees rotation
            label.shift(DOWN * 0.2)
            label.shift(LEFT * 0.1)


        # Create an arrow from the reuse_example to the chart
        right_of_chart = chart.get_right()
        right_of_chart[1] = reuse_example.get_left()[1]
        with self.voiceover(text="Which then result in up to 6x speedup per solve!") as tracker:
            arrow = Arrow(reuse_example.get_left(), right_of_chart, color=WHITE)
            self.play(Create(arrow))

            # Animate the chart
            self.play(
                    chart.animate.change_bar_values(final_values),
                    run_time=0.5)
        


