from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService


class Section2():
    def __init__(self, scene: Scene | VoiceoverScene):
        self.scene = scene
        self.FadeIn_time = 0.5
        self.second_order_block_time = 0.5
        self.transition_to_seq_block_time = 0.5
        self.brace_time = 0.2
        self.arrow_time = 0.5
        self.section2_planned_time = 8
        self.section2_total_time = 0

        # Section 2 runtime
        self.parth_integration_block = 2.5
        self.parth_reuse_example = 0.5
        self.reuse_delay = 0.7
        self.shift_reuse_example = 1.5
        self.speedup_chart_time = 0.5
        self.chart_wait_time = 1
        
    # Section 2
    def _parth_3_lines_integration(self) -> Code:
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

    def _parth_intergration_block(self, font_size=14, container_width=5, container_height=4) -> VGroup:
        # create a big transparent rectangle with a border
        # within that rectangle, there is a figure  (Figures/section2/brain.svg) on top with three arrow comming out of it to 3 other
        # rectangles withing that block. The name of each rectangle is Accelerate, CHOLMOD, and MKL Pardiso

        # Create the main container rectangle
        container = Rectangle(
            width=container_width, 
            height=container_height, 
            stroke_color=WHITE, 
            stroke_width=2, 
            fill_opacity=0.2,
            fill_color=WHITE
        )
        
        # Load the brain SVG and position it on the middle left side
        border_margin_scale = 0.05
        brain = SVGMobject("Figures/section2/adapt.svg", use_svg_cache=True)
        brain.scale_to_fit_width(container_width / 5)
        brain.move_to(container.get_left() + RIGHT * brain.get_width() / 2 + RIGHT * border_margin_scale * container_width)
        #Add a subtitle to the brain
        brain_subtitle = Text("Parth", font_size=font_size, color=WHITE)
        brain_subtitle.scale_to_fit_width(brain.get_width() * 0.9)
        brain_subtitle.move_to(brain.get_bottom() + DOWN * 0.5)
        
        # Create three rectangles for the libraries in a column on the right side
        rect_height = container_height / 4
        rect_width = container_width / 2.2
        accelerate_rect = Rectangle(
            width=rect_width, 
            height=rect_height, 
            stroke_color=WHITE, 
            stroke_width=1.5,
            fill_opacity=0.3,
            fill_color=GREEN
        )

        cholmod_rect = Rectangle(
            width=rect_width, 
            height=rect_height, 
            stroke_color=WHITE, 
            stroke_width=1.5,
            fill_opacity=0.3,
            fill_color=YELLOW
        )
        
        mkl_rect = Rectangle(
            width=rect_width, 
            height=rect_height, 
            stroke_color=WHITE, 
            stroke_width=1.5,
            fill_opacity=0.4,
            fill_color=RED
        )
        
        #Arrange the rectangles in a column
        rectangles = VGroup(accelerate_rect, cholmod_rect, mkl_rect)
        rectangles.arrange(DOWN, buff=0.1 * rect_height)
        rectangles.move_to(container.get_right() + LEFT * rectangles.width / 2 + LEFT * border_margin_scale * container_width)

        # Add text labels to the rectangles
        accelerate_text = Text("Accelerate", font_size=font_size, color=WHITE)
        accelerate_text.scale_to_fit_width(rect_width * 0.9)
        accelerate_text.move_to(accelerate_rect.get_center())
        
        cholmod_text = Text("CHOLMOD", font_size=font_size, color=WHITE)
        cholmod_text.scale_to_fit_width(rect_width * 0.9)
        cholmod_text.move_to(cholmod_rect.get_center())
        
        mkl_text = Text("MKL Pardiso", font_size=font_size, color=WHITE)
        mkl_text.scale_to_fit_width(rect_width * 0.9)
        mkl_text.move_to(mkl_rect.get_center())
        
        # Create arrows from brain to each rectangle (horizontal arrows)
        arrow1 = Arrow(
            brain.get_right() + RIGHT * border_margin_scale * container_width,
            accelerate_rect.get_left() + LEFT * border_margin_scale * container_width,
            buff=0.1,
            stroke_width=2,
            color=WHITE
        )
        
        arrow2 = Arrow(
            brain.get_right() + RIGHT * border_margin_scale * container_width,
            cholmod_rect.get_left() + LEFT * border_margin_scale * container_width,
            buff=0.1,
            stroke_width=2,
            color=WHITE
        )
        
        arrow3 = Arrow(
            brain.get_right() + RIGHT * border_margin_scale * container_width,
            mkl_rect.get_left() + LEFT * border_margin_scale * container_width,
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
    
    def _run_scene(self):
        integration_block_to_reuse_block_arrow = None
        integration_block = self._parth_intergration_block(font_size=32)
        integration_block.to_edge(LEFT)

        self.scene.play(Create(integration_block), run_time=self.parth_integration_block)

        reuse_example = self._parth_reuse_example()
        reuse_example.scale_to_fit_width(integration_block.get_width())
        reuse_example.to_edge(RIGHT)
        # Draw an arrow from the integration_block to the reuse_example
        integration_block_to_reuse_block_arrow = Arrow(integration_block.get_right(), reuse_example.get_left(), color=WHITE)
        self.scene.play(FadeIn(reuse_example), run_time=self.FadeIn_time)
        self.scene.play(Create(integration_block_to_reuse_block_arrow), run_time=self.arrow_time)
        self.scene.wait(self.reuse_delay)
        
        # Fade out the integration_block and move the reuse_example to the left
        self.scene.play(FadeOut(integration_block), FadeOut(integration_block_to_reuse_block_arrow), reuse_example.animate.to_edge(LEFT), run_time=self.shift_reuse_example)
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
        chart.to_edge(RIGHT)
        
        
        # Manually rotate the x-axis labels by 45 degrees
        for label in chart.x_axis.labels:
            label.rotate(PI/4)  # 45 degrees rotation
            label.shift(DOWN * 0.2)
            label.shift(LEFT * 0.1)


        # Create an arrow from the reuse_example to the chart
        left_of_The_chart = chart.get_left()
        left_of_The_chart[1] = reuse_example.get_left()[1]

        accelerate_speedup = DecimalNumber(0, num_decimal_places=1)
        cholmod_speedup = DecimalNumber(0, num_decimal_places=1)
        mkl_speedup = DecimalNumber(0, num_decimal_places=1)

        Accelerate_speedup_symbol = Tex(r"$\times$", font_size=22)
        CHOLMOD_speedup_symbol = Tex(r"$\times$", font_size=22)
        MKL_speedup_symbol = Tex(r"$\times$", font_size=22)

        def update_speedup_symbols(bar: BarChart):
            Accelerate_speedup_symbol.next_to(accelerate_speedup, RIGHT, buff=0)
            CHOLMOD_speedup_symbol.next_to(cholmod_speedup, RIGHT, buff=0)
            MKL_speedup_symbol.next_to(mkl_speedup, RIGHT, buff=0)

        def chart_updater(bar: BarChart):
            chart.change_bar_values([accelerate_speedup.get_value(),
            cholmod_speedup.get_value(), mkl_speedup.get_value()])
            
            accelerate_speedup.next_to(chart.bars[0], UP, buff=0.1)
            cholmod_speedup.next_to(chart.bars[1], UP, buff=0.1)
            mkl_speedup.next_to(chart.bars[2], UP, buff=0.1)
                    
            Accelerate_speedup_symbol.next_to(accelerate_speedup, RIGHT, buff=0.05)
            CHOLMOD_speedup_symbol.next_to(cholmod_speedup, RIGHT, buff=0.05)
            MKL_speedup_symbol.next_to(mkl_speedup, RIGHT, buff=0.05)
        
        
        Accelerate_speedup_symbol.add_updater(update_speedup_symbols, call_updater=True)
        CHOLMOD_speedup_symbol.add_updater(update_speedup_symbols, call_updater=True)
        MKL_speedup_symbol.add_updater(update_speedup_symbols, call_updater=True)

        arrow = Arrow(reuse_example.get_right(), left_of_The_chart, color=WHITE)
        self.scene.play(Create(arrow), runtime=self.arrow_time)
        
        
        chart.add_updater(chart_updater, call_updater=True)
        self.scene.add(chart, accelerate_speedup, cholmod_speedup, mkl_speedup, Accelerate_speedup_symbol, CHOLMOD_speedup_symbol, MKL_speedup_symbol)

        # Animate the chart
        self.scene.play(
                ChangeDecimalToValue(accelerate_speedup, final_values[0]),
                ChangeDecimalToValue(cholmod_speedup, final_values[1]),
                ChangeDecimalToValue(mkl_speedup, final_values[2]), runtime=self.speedup_chart_time)
        self.scene.wait(self.chart_wait_time)

    def section2(self):
        if isinstance(self.scene, VoiceoverScene):
            with self.scene.voiceover(text="With just 3 lines of code, integrating Parth into Cholesky solvers enables adaptive sparsity analysis reuse, achieving up to 5.9× speedup per solves!") as tracker:
                self._run_scene()
        else:
            self._run_scene()   
