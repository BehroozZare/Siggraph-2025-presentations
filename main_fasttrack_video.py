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
            
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.3)

        return Group(*lines)
    
    def application_block(self, block_total_width: float, block_total_height: int, font_size: int) -> VGroup:
        # Create a box with Second-order Optimizer in it
        optimizer_block = Rectangle(width=block_total_width, height=block_total_height,
                                       stroke_color=WHITE, stroke_width=1.5, fill_opacity=0.3, fill_color=YELLOW_A)
        optimizer_block_text = Text("Second-order \n Optimizer", font_size=font_size, color=WHITE)
        #Put the optimizer_block_text inside the optimizer_block
        optimizer_block_text.scale_to_fit_width(optimizer_block.get_width() * 0.8)
        optimizer_block.stretch_to_fit_height(optimizer_block_text.get_height()* 1.1)
        optimizer_block_text.move_to(optimizer_block.get_center())

        return VGroup(optimizer_block, optimizer_block_text)

    def seq_block(self, num_seq_lines: int, font_size: int) -> VGroup:
        # Create a sequence of linear systems using a for loop
        seq_equations = []
        for i in range(1, num_seq_lines + 1):  # Creates seq1 through seq6
            seq_equations.append(Tex(f"$H_{{{i}}} d_{{{i}}} = -g_{{{i}}}$", font_size=font_size))

        # Put all the seq_equations into a VGroup
        seq_equations_group = VGroup(*seq_equations)
        seq_equations_group.arrange(DOWN, buff=0.1)

        return seq_equations_group
        
    
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
        NUM_SEQ_LINES = 5
        application_block = self.application_block(block_total_width=2, block_total_height=1, font_size=24)
        application_block.to_edge(LEFT, buff=1)
        seq_block = self.seq_block(num_seq_lines=NUM_SEQ_LINES, font_size=24)
        seq_block.move_to(application_block.get_center())
        self.play(FadeIn(application_block), runtime=0.1)
        # Using the algorithm block to show an example of a sequence of linear systems
        # application_block = self.make_algorithm_block()

        # Position the algorithm block below the explanation
        # algorithm_block.next_to(explanation, DOWN, buff=2)
        # Start with what is important for us
        with self.voiceover(text="Applications like second-order optimization require back-to-back sparse Cholesky solves,") as tracker:    
            self.wait(2)
            self.play(Transform(application_block, seq_block), runtime=0.5)
            #Brace the algorithm block
            application_block_brace = BraceLabel(application_block, text=r"\text{Search Direction Computation}", buff=0.1, font_size=22)
            application_block_brace.next_to(application_block, DOWN, buff=0.1)
            self.add(application_block_brace)
            self.play(FadeIn(application_block_brace))


        with self.voiceover(text="but dynamic sparsity imposes high analysis overhead!") as tracker:
            # Create the matrix with subtitle (iteration 1)
            
            # Add a surrounding box around the first linear system in the application_block
            box = SurroundingRectangle(application_block[0], color=RED, buff=0.1)

            matrix_font_size = 32
            matrix_group = self.make_matrix_with_subtitle(matrix_font_size, 1, sparsity=0.1)
            matrix_group.next_to(application_block, RIGHT, buff=2)
            
            matrix_group_brace = BraceLabel(matrix_group, text=r"\text{Sparse Cholesky Solve}", buff=0.1, font_size=22)
            matrix_group_brace.next_to(matrix_group, DOWN, buff=0.1)

            init_values = [0, 0]
            final_values = [84, 16]
            symbolic_val = DecimalNumber(0, num_decimal_places=0)
            numeric_val = DecimalNumber(0, num_decimal_places=0)

            symbolic_percent = Tex("\%", font_size=22)
            numeric_percent = Tex("\%", font_size=22)
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
                        # Show the matrix
                                    #Create an arrow from the box to the matrix_group
            box_to_matrix_arrow = Arrow(box.get_right(), matrix_group.get_left(), color=WHITE)

            # # Manually rotate the x-axis labels by 45 degrees
            # for label in chart.x_axis.labels:
            #     # label.rotate(PI/4)  # 45 degrees rotation
            #     label.shift(DOWN * 0.2)

            
            # Animate through different iterations with changing sparsity
            # Calculate the deceleration values for the gauge
            start_iter = 1
            end_iter = NUM_SEQ_LINES
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
            chart_brace = BraceLabel(chart, text=r"\text{Sparsity Analysis Overhead}", buff=0.1, font_size=22)
            
            self.play(FadeIn(matrix_group), FadeIn(matrix_group_brace), FadeIn(box),
                    FadeIn(chart), FadeIn(chart_brace), FadeIn(symbolic_percent), FadeIn(numeric_percent), run_time=0.5)
            self.play(Create(box_to_matrix_arrow), runtime=0.1)



            # Create new matrix with different sparsity
            total_animation_time = tracker.get_remaining_duration() + 1
            animation_time_per_iteration = total_animation_time / num_iterations
            for i, iteration in enumerate(range(start_iter, end_iter)):
                new_matrix_group = self.make_matrix_with_subtitle(matrix_font_size, iteration + 1, sparsity=0.1)
                new_matrix_group.move_to(matrix_group.get_center())
                new_box = SurroundingRectangle(application_block[iteration], color=RED, buff=0.1)
                new_arrow = Arrow(new_box.get_right(), new_matrix_group.get_left(), color=WHITE)
                new_matrix_group_brace = BraceLabel(new_matrix_group, text=r"\text{Sparse Cholesky Solve}", buff=0.1, font_size=22)
                new_transform_group = VGroup(new_box, new_arrow, new_matrix_group, new_matrix_group_brace)
                old_transform_group = VGroup(box, box_to_matrix_arrow, matrix_group, matrix_group_brace)

                # Animate both matrix transformation and gauge deceleration simultaneously
                if i < end_iter - 3:
                    self.play(
                        Transform(old_transform_group, new_transform_group),
                        ChangeDecimalToValue(symbolic_val, final_values[0] / num_iterations * (i + 1)),
                        ChangeDecimalToValue(numeric_val, final_values[1] / num_iterations * (i + 1)),
                        run_time=animation_time_per_iteration
                    )
                else:
                    #Making the inspector color in bar chart bold and moving
                    inspector_bar = chart.bars[0]
                    inspector_bar.set_color(RED)
                    inspector_bar.set_weight(BOLD)
                    inspector_label = chart.x_axis.labels[0]
                    conclusion = Tex(r"Sparsity analysis is \textcolor{red}{expensive}!", font_size=32)
                    conclusion.to_edge(UP, buff=1)
                    self.play(
                        Transform(old_transform_group, new_transform_group),
                        ChangeDecimalToValue(symbolic_val, final_values[0] / num_iterations * (i + 1)),
                        ChangeDecimalToValue(numeric_val, final_values[1] / num_iterations * (i + 1)),
                        Write(conclusion),
                        run_time=animation_time_per_iteration
                    )
    
            


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

    def section2(self):
        integration_block_to_reuse_block_arrow = None
        with self.voiceover(text="By integrating Parth into Cholesky solvers with just 3 lines of code...") as tracker:
            integration_block = self._parth_intergration_block(font_size=32)
            integration_block.to_edge(LEFT)

            self.play(Create(integration_block), run_time=tracker.duration)

        with self.voiceover(text="we allow for adaptive reuse of sparsity analysis,") as tracker:
            reuse_example = self._parth_reuse_example()
            reuse_example.scale_to_fit_width(integration_block.get_width())
            reuse_example.to_edge(RIGHT)
            # Draw an arrow from the integration_block to the reuse_example
            integration_block_to_reuse_block_arrow = Arrow(integration_block.get_right(), reuse_example.get_left(), color=WHITE)
            self.play(FadeIn(reuse_example), run_time=tracker.get_remaining_duration() - 0.5)
            self.play(Create(integration_block_to_reuse_block_arrow), run_time=0.5)
            
        
        with self.voiceover(text="which allows for up to 5.9x speedup per solve!") as tracker:
            # Fade out the integration_block and move the reuse_example to the left
            self.play(FadeOut(integration_block), FadeOut(integration_block_to_reuse_block_arrow), reuse_example.animate.to_edge(LEFT))
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
            self.play(Create(arrow), runtime=0.1)
            
            self.add(accelerate_speedup, cholmod_speedup, mkl_speedup, Accelerate_speedup_symbol, CHOLMOD_speedup_symbol, MKL_speedup_symbol)
            chart.add_updater(chart_updater, call_updater=True)
            self.add(chart)

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
            # QR code link
            qr_code = ImageMobject("Figures/section3/qr_code.png")
            qr_code.scale_to_fit_width(paper_title.get_width() * 0.2)
            qr_code.next_to(paper_title, DOWN, buff=1)

        
            # Show elements
            self.play(FadeIn(paper_title), FadeIn(qr_code), runtime=tracker.duration)

    def construct(self):
        # set your TTS engine once
        # self.set_speech_service(GTTSService(speed=1.5))
        # self.set_speech_service(GTTSService(slow=True))
        self.set_speech_service(RecorderService())
        # — Section 1 —
        self.section1()
        #Clear the scene
        if self.mobjects:
            self.play(FadeOut(*self.mobjects), run_time=0.5)
            self.wait(0.2)
            

        # — Section 2 —
        self.section2()
        if self.mobjects:
            self.play(FadeOut(*self.mobjects), run_time=0.3)
            self.wait(0.1)

        #— Section 3 —
        self.section3()
        if self.mobjects:
            self.play(FadeOut(*self.mobjects), run_time=1)