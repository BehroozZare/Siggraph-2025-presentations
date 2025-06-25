from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService


# 2) extend the pre-amble
template = TexTemplate()
template.add_to_preamble(r"\usepackage{xcolor}")

config.tex_template = template


class FastTrackComplete(VoiceoverScene):
    def make_title(self) -> Tex:
        title = Tex(r"Adaptive Algebraic Reuse of Reordering in Cholesky Factorizations with Dynamic Sparsity Patterns", font_size=36)
        return title
    
    def make_importancce_text(self) -> Tex:
        text = Tex(r"Fast execution of a \textcolor{red}{sequence} of \textcolor{green}{Sparse Cholesky solves} is important in many graphics and scientific applications!", 
                          font_size=36, color=BLUE)
        return text
    
    def make_algorithm_block(self) -> VGroup:
        template = TexTemplate()
        template.add_to_preamble(r"\usepackage{algorithmic}")
        template.add_to_preamble(r"\usepackage{xcolor}")
        entries = [
            (0, r"\textbf{while} not converged:"),
            (1, r"$g \gets \nabla f(x)$"),
            (1, r"$H \gets \nabla^2 f(x)$"),
            (1, r"Solve $H \cdot d = -g$"),
            (1, r"$x \gets x + \alpha \cdot d$"),
            (1, r"Check convergence: $\|g\| < \epsilon$"),
        ]
        lines = []
        for i, (indent, txt) in enumerate(entries):
            line = rf"{i+1}.\quad" + r"\quad"*indent + " " + txt
            lines.append(Tex(line, tex_template=template, font_size=32))
        return Group(*lines)
    
    def make_linear_system_seq(self) -> Group:
        # Create a sequence of linear systems
        sequence = Tex(r"$A_1 x_1 = b_1 \rightarrow A_2 x_2 = b_2 \rightarrow \dots \rightarrow A_i x_i = b_i \rightarrow \dots \rightarrow A_n x_n = b_n$", font_size=36)
        return sequence

    def make_random_matrix(self) -> Matrix:
        # Create a random matrix
        matrix = np.random.randn(10, 10)
        return matrix

    def make_matrix_with_subtitle(self, font_size: int, iteration: int, sparsity: float = 0.2) -> VGroup:
        # Create a symmetric sparse matrix pattern
        size = 16  # Keep size manageable for LaTeX
        
        # Create a random matrix
        np.random.seed(iteration * 100 + int(sparsity * 1000))
        matrix_data = np.random.randn(size, size)
        
        # Make it symmetric by averaging with its transpose
        matrix_data = (matrix_data + matrix_data.T) / 2
        
        # Ensure diagonal entries are always non-zero (positive for stability)
        np.fill_diagonal(matrix_data, np.abs(np.random.randn(size)) + 0.1)
        
        # Make it sparse by setting some OFF-DIAGONAL elements to zero
        # Create mask for off-diagonal elements only
        off_diag_mask = ~np.eye(size, dtype=bool)  # True for off-diagonal elements
        sparsity_mask = np.random.random((size, size)) > sparsity
        # Apply sparsity only to off-diagonal elements
        matrix_data[off_diag_mask & sparsity_mask] = 0
        
        # Ensure symmetry is maintained after sparsification
        matrix_data = np.triu(matrix_data) + np.triu(matrix_data, 1).T
        
        # Create the LaTeX matrix string with proper formatting
        # Create column alignment string
        col_align = "|" + "c" * size + "|"
        
        matrix_str = r"$\begin{array}{" + col_align + r"}"
        for i in range(size):
            row_parts = []
            for j in range(size):
                if abs(matrix_data[i, j]) > 1e-10:  # Non-zero element
                    row_parts.append(r"\ast")
                else:  # Zero element
                    row_parts.append("")  # Empty string for zero elements
            matrix_str += " & ".join(row_parts)
            if i < size - 1:
                matrix_str += r" \\ "
        matrix_str += r"\end{array}$"
        
        # Create the matrix as LaTeX
        matrix = Tex(matrix_str, font_size=font_size)  # Reduced font size for better fit
        matrix.scale(0.5)  # Scale down more for better visibility
        
        # Create subtitle
        subtitle = Tex(f"$H_{{{iteration}}}$", font_size=font_size, color=RED)
        
        # Group matrix and subtitle
        group = VGroup(matrix, subtitle)
        group.arrange(DOWN, buff=0.3)
        
        return group

    def section1(self):
        # Start with what is important for us
        with self.voiceover(text="Applications like second-order optimization involve back-to-back sparse Cholesky solves, but fast-changing sparsity makes each analysis extremely expensive!") as tracker:
            # explanation = Tex(r"Applications like second-order optimization rely on \textcolor{red}{back-to-back} sparse Cholesky solves, but \textcolor{green}{fast-changing sparsity} makes each analysis prohibitively expensive!", 
            #                 font_size=36, color=BLUE)
            # explanation.to_edge(UP, buff=1)
            # self.play(Write(explanation))


            # Using the algorithm block to show an example of a sequence of linear systems
            algorithm_block = self.make_algorithm_block()
            algorithm_block.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            
            # Position the algorithm block below the explanation
            # algorithm_block.next_to(explanation, DOWN, buff=2)
            algorithm_block.move_to(ORIGIN)
            algorithm_block.to_edge(LEFT)
            self.add(algorithm_block)
            #Brace the algorithm block
            algorithm_block_brace = BraceLabel(algorithm_block, text="Second-order Optimization", buff=0.1, font_size=22)
            algorithm_block_brace.next_to(algorithm_block, DOWN, buff=0.1)
            self.add(algorithm_block_brace)
            self.play(FadeIn(algorithm_block), FadeIn(algorithm_block_brace))

            # Add a surrounding box around the algorithm_block[3]
            box = SurroundingRectangle(algorithm_block[3], color=RED, buff=0.1)
            self.play(Create(box))
            
            # Create the matrix with subtitle (iteration 1)
            matrix_font_size = 32
            matrix_group = self.make_matrix_with_subtitle(matrix_font_size, 1, sparsity=0.1)
            matrix_group.next_to(algorithm_block, RIGHT, buff=0.5)
            # self.add(matrix_group)
            matrix_group_brace = BraceLabel(matrix_group, text="Sparse Cholesky Solve", buff=0.1, font_size=22)
            matrix_group_brace.next_to(matrix_group, DOWN, buff=0.1)
            # self.add(matrix_group_brace)
            # matrix_group.move_to(ORIGIN)
            
            # Create arrow from the Hd=-g line to the matrix
            arrow = Arrow(
                algorithm_block[3].get_right() + RIGHT * 0.1,
                matrix_group.get_left() + LEFT * 0.2,
                color=YELLOW,
                buff=0.1
            )
            # First draw the arrow
            self.play(Create(arrow), run_time=0.01)
            
            # Show the matrix
            self.play(FadeIn(matrix_group), FadeIn(matrix_group_brace), run_time=0.5)
            
            init_values = [0, 0]
            final_values = [84, 16]
            symbolic_val = DecimalNumber(0, num_decimal_places=0)
            numeric_val = DecimalNumber(0, num_decimal_places=0)
            self.add(symbolic_val, numeric_val)

            symbolic_percent = Tex("\%", font_size=22)
            numeric_percent = Tex("\%", font_size=22)
            self.add(symbolic_percent, numeric_percent)
            # Create the bar chart that shows the inspector bottleneck
            chart = BarChart(
                values=init_values,
                bar_names=["Symbolic", "Numeric"],
                y_range=[0, 100, 20],
                y_length=2,
                x_length=3,
                x_axis_config={"font_size": 22},
                y_axis_config={"font_size": 22, "decimal_number_config": {"unit": "\\%", "num_decimal_places": 0}}
            )
            chart.next_to(matrix_group, RIGHT, buff=0.5)
            self.add(chart)

            # # Manually rotate the x-axis labels by 45 degrees
            # for label in chart.x_axis.labels:
            #     # label.rotate(PI/4)  # 45 degrees rotation
            #     label.shift(DOWN * 0.2)

            
            # Animate through different iterations with changing sparsity
            # Calculate the deceleration values for the gauge
            start_iter = 2
            end_iter = 8
            num_iterations = end_iter - start_iter
            
            def update_percent_symbols(bar: BarChart):
                symbolic_percent.next_to(symbolic_val, RIGHT, buff=0.05)
                numeric_percent.next_to(numeric_val, RIGHT, buff=0.05)
    
            def chart_updater(bar: BarChart):
                sym_bar, num_bar = chart.bars
                chart.change_bar_values([symbolic_val.get_value(), numeric_val.get_value()])
                symbolic_val.next_to(sym_bar, UP, buff=0.1)
                numeric_val.next_to(num_bar, UP, buff=0.1)

            chart.add_updater(chart_updater, call_updater=True)
            symbolic_percent.add_updater(update_percent_symbols, call_updater=True)
            numeric_percent.add_updater(update_percent_symbols, call_updater=True)
            chart_brace = BraceLabel(chart, text="Sparsity Analysis Overhead", buff=0.1, font_size=22)
            self.add(chart_brace)



            for i, iteration in enumerate(range(start_iter, end_iter)):
                # Create new matrix with different sparsity
                new_matrix_group = self.make_matrix_with_subtitle(matrix_font_size, iteration, sparsity=0.1)
                new_matrix_group.move_to(matrix_group.get_center())

                # Animate both matrix transformation and gauge deceleration simultaneously
                self.play(
                    Transform(matrix_group, new_matrix_group),
                    ChangeDecimalToValue(symbolic_val, final_values[0] / num_iterations * (i + 1)),
                    ChangeDecimalToValue(numeric_val, final_values[1] / num_iterations * (i + 1)),
                    run_time=0.7
                )

                
            #Making the inspector color in bar chart bold and moving
            inspector_bar = chart.bars[0]
            inspector_bar.set_color(RED)
            inspector_bar.set_weight(BOLD)
            inspector_label = chart.x_axis.labels[0]
            original_color = inspector_label.color


            
            self.play(inspector_bar.animate.shift(UP * 0.1), run_time=0.2)
            self.play(inspector_bar.animate.shift(DOWN * 0.1), inspector_label.animate.set_color(RED).shift(DOWN * 0.1), run_time=0.2)

            remaining_time = tracker.get_remaining_duration();
            conclusion = Tex(r"Sparsity analysis is \textcolor{red}{expensive}!", font_size=32)
            conclusion.to_edge(UP, buff=1)
            self.play(Write(conclusion), run_time=remaining_time)

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

    def section2(self):
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

        accelerate_speedup = DecimalNumber(0, num_decimal_places=1)
        cholmod_speedup = DecimalNumber(0, num_decimal_places=1)
        mkl_speedup = DecimalNumber(0, num_decimal_places=1)
        self.add(accelerate_speedup, cholmod_speedup, mkl_speedup)
        
        Accelerate_speedup_symbol = Tex(r"$\times$", font_size=22)
        CHOLMOD_speedup_symbol = Tex(r"$\times$", font_size=22)
        MKL_speedup_symbol = Tex(r"$\times$", font_size=22)
        self.add(Accelerate_speedup_symbol, CHOLMOD_speedup_symbol, MKL_speedup_symbol)

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
        

        chart.add_updater(chart_updater, call_updater=True)
        Accelerate_speedup_symbol.add_updater(update_speedup_symbols, call_updater=True)
        CHOLMOD_speedup_symbol.add_updater(update_speedup_symbols, call_updater=True)
        MKL_speedup_symbol.add_updater(update_speedup_symbols, call_updater=True)


        self.add(chart)

        with self.voiceover(text="Which then result in up to 6x speedup per solve!") as tracker:
            arrow = Arrow(reuse_example.get_left(), right_of_chart, color=WHITE)
            self.play(Create(arrow), runtime=0.1)

            # Animate the chart
            self.play(
                    ChangeDecimalToValue(accelerate_speedup, final_values[0]),
                    ChangeDecimalToValue(cholmod_speedup, final_values[1]),
                    ChangeDecimalToValue(mkl_speedup, final_values[2]))
        
    def _parth_paper_title(self):
        title = Tex(r"Adaptive Algebraic Reuse of Reordering in Cholesky Factorizations with Dynamic Sparsity Patterns", font_size=36)
        return title
    
    def _parth_site(self):
        subtitle = Tex(r"github.com/behrooz/parth",  font_size=36, color=BLUE)
        return subtitle
    
    def section3(self):
        # Main title
        with self.voiceover(text="Thank you and see you in our session!") as tracker:
            paper_title = self._parth_paper_title()
            paper_title.to_edge(UP, buff=1)
        
            # Subtitle
            paper_site = self._parth_site()
            paper_site.next_to(paper_title, DOWN, buff=0.5)
        
            # Show elements
            self.play(Write(paper_title))
            self.play(Write(paper_site))

    def construct(self):
        # set your TTS engine once
        # self.set_speech_service(GTTSService(speed=1.5))
        self.set_speech_service(GTTSService(slow=True))
        # — Section 1 —
        self.section1()
        #Clear the scene
        if self.mobjects:
            self.play(FadeOut(*self.mobjects), run_time=0.3)
            self.wait(0.1)

        # — Section 2 —
        self.section2()
        if self.mobjects:
            self.play(FadeOut(*self.mobjects), run_time=0.3)
            self.wait(0.1)

        # — Section 3 —
        self.section3()
        if self.mobjects:
            self.play(FadeOut(*self.mobjects), run_time=0.3)
            self.wait(0.1)