from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService


# 2) extend the pre-amble
template = TexTemplate()
template.add_to_preamble(r"\usepackage{xcolor}")

config.tex_template = template


# Section1: Handles the animation and logic for the problem definition section of the FastTrack video.
class Section1():
    def __init__(self, scene: Scene | VoiceoverScene):
        # Initialize the section with timing and animation parameters
        self.scene = scene
        self.FadeIn_time = 0.5  # Time for fade-in animations
        self.second_order_block_time = 0.5  # Time for second-order block animation
        self.transition_to_seq_block_time = 0.5  # Time for transitioning to sequence block
        self.brace_time = 0.2  # Time for brace animation
        self.arrow_time = 0.5  # Time for arrow animation
        self.per_solve_time = 0.5  # Time per solve iteration
        self.total_number_of_iterations = 5  # Number of iterations for the sequence
        self.section1_planned_time = 8  # Planned total time for section 1
        self.section1_total_time = 0  # Actual total time (computed below)

        # Section 1 runtime -> 8 seconds
        self.second_order_block_time = 0.5
        self.second_order_block_time - 0.1 > 0, "Second order block time is too short"
        
        self.transition_to_seq_block_time = 0.5
        self.brace_time = 0.2
        assert self.transition_to_seq_block_time - self.brace_time > 0, "Transition to seq block time is too short"

        self.surronding_box_time = 0.2  # Time for surrounding box animation
        self.matrix_and_chart_appear_time = 0.5  # Time for matrix and chart appearance
        self.total_number_of_iterations = 5
        self.per_solve_time = 0.5
        self.emoji_time = 0.2  # Time for emoji animation

        # Calculate total time for section 1
        self.section1_total_time = self.second_order_block_time + self.transition_to_seq_block_time + self.surronding_box_time + self.matrix_and_chart_appear_time + self.arrow_time + self.per_solve_time * self.total_number_of_iterations + self.emoji_time
        
    def make_title(self) -> Tex:
        # Create the title for the section
        title = Tex(r"Adaptive Algebraic Reuse of Reordering in Cholesky Factorizations with Dynamic Sparsity Patterns", font_size=36)
        return title
    
    def make_importancce_text(self) -> Tex:
        # Create the importance text for the section
        text = Tex(r"Fast execution of a \textcolor{red}{sequence} of \textcolor{green}{Sparse Cholesky solves} is important in many graphics and scientific applications!", 
                          font_size=36, color=BLUE)
        return text
    
    def make_algorithm_block(self) -> VGroup:
        # Create a block showing the Newton algorithm steps
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
            # Indent and format each algorithm step
            line = rf"{i+1}.\quad" + r"\quad"*indent + " " + txt
            lines.append(Tex(line, tex_template=template, font_size=32))
            
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.3)

        return Group(*lines)
    
    def application_block(self, block_total_width: float, block_total_height: int, font_size: int) -> VGroup:
        # Create a visual block representing the second-order optimizer
        optimizer_block = Rectangle(width=block_total_width, height=block_total_height,
                                       stroke_color=WHITE, stroke_width=1.5, fill_opacity=0.3, fill_color=YELLOW_A)
        optimizer_block_text = Text("Second-order \n Optimizer", font_size=font_size, color=WHITE)
        # Fit the text inside the block
        optimizer_block_text.scale_to_fit_width(optimizer_block.get_width() * 0.8)
        optimizer_block.stretch_to_fit_height(optimizer_block_text.get_height()* 1.1)
        optimizer_block_text.move_to(optimizer_block.get_center())

        return VGroup(optimizer_block, optimizer_block_text)

    def seq_block(self, num_seq_lines: int, font_size: int) -> VGroup:
        # Create a sequence of linear system equations for the animation
        seq_equations = []
        for i in range(1, num_seq_lines + 1):  # Creates seq1 through seq6
            seq_equations.append(Tex(f"$H_{{{i}}} d_{{{i}}} = -g_{{{i}}}$", font_size=font_size))

        # Group and arrange the equations vertically
        seq_equations_group = VGroup(*seq_equations)
        seq_equations_group.arrange(DOWN, buff=0.1)

        return seq_equations_group
        
    
    def make_linear_system_seq(self) -> Group:
        # Create a LaTeX string representing a sequence of linear systems
        sequence = Tex(r"$A_1 x_1 = b_1 \rightarrow A_2 x_2 = b_2 \rightarrow \dots \rightarrow A_i x_i = b_i \rightarrow \dots \rightarrow A_n x_n = b_n$", font_size=36)
        return sequence

    def make_random_matrix(self) -> Matrix:
        # Generate a random 10x10 matrix (for demonstration)
        matrix = np.random.randn(10, 10)
        return matrix

    def make_matrix_with_subtitle(self, font_size: int, iteration: int, sparsity: float = 0.2) -> VGroup:
        # Create a symmetric sparse matrix pattern for visualization
        size = 16  # Keep size manageable for LaTeX
        
        # Seed for reproducibility based on iteration and sparsity
        np.random.seed(iteration * 100 + int(sparsity * 1000))
        matrix_data = np.random.randn(size, size)
        
        # Make the matrix symmetric
        matrix_data = (matrix_data + matrix_data.T) / 2
        
        # Ensure diagonal entries are always non-zero (positive for stability)
        np.fill_diagonal(matrix_data, np.abs(np.random.randn(size)) + 0.1)
        
        # Make it sparse by setting some OFF-DIAGONAL elements to zero
        off_diag_mask = ~np.eye(size, dtype=bool)  # True for off-diagonal elements
        sparsity_mask = np.random.random((size, size)) > sparsity
        matrix_data[off_diag_mask & sparsity_mask] = 0
        
        # Ensure symmetry is maintained after sparsification
        matrix_data = np.triu(matrix_data) + np.triu(matrix_data, 1).T
        
        # Create the LaTeX matrix string for visualization
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
        matrix.scale(0.5)  # Scale down for better visibility
        
        # Create subtitle for the matrix
        subtitle = Tex(f"$H_{{{iteration}}}$", font_size=font_size, color=RED)
        
        # Group matrix and subtitle
        group = VGroup(matrix, subtitle)
        group.arrange(DOWN, buff=0.3)
        
        return group
    
    def run_scene(self):
        # Main animation logic for section 1
        application_block = self.application_block(block_total_width=2, block_total_height=1, font_size=24)
        application_block.to_edge(LEFT, buff=1)
        seq_block = self.seq_block(num_seq_lines=self.total_number_of_iterations, font_size=24)
        seq_block.move_to(application_block.get_center())

        # Animate the appearance of the application block
        self.scene.play(FadeIn(application_block), runtime=self.FadeIn_time)
        self.scene.wait(self.second_order_block_time - 0.1)
        self.scene.play(Transform(application_block, seq_block), runtime=self.transition_to_seq_block_time - self.brace_time)
        # Add a brace label to the algorithm block
        application_block_brace = BraceLabel(application_block, text=r"\text{Search Direction Computation}", buff=0.1, font_size=22)
        application_block_brace.next_to(application_block, DOWN, buff=0.1)
        self.scene.play(FadeIn(application_block_brace), runtime=self.FadeIn_time)

        # Add a surrounding box around the first linear system in the application_block
        box = SurroundingRectangle(application_block[0], color=RED, buff=0.1)

        # Create the matrix block for visualization
        matrix_font_size = 32
        matrix_group = self.make_matrix_with_subtitle(matrix_font_size, 1, sparsity=0.1)
        matrix_group.next_to(application_block, RIGHT, buff=2)
        
        # Add a brace label to the matrix group
        matrix_group_brace = BraceLabel(matrix_group, text=r"\text{Sparse Cholesky Solve}", buff=0.1, font_size=22)
        matrix_group_brace.next_to(matrix_group, DOWN, buff=0.1)

        # Create the arrow between the sequence box and the matrix block
        box_to_matrix_arrow = Arrow(box.get_right(), matrix_group.get_left(), color=WHITE)

        # Create the bar chart and related elements for sparsity analysis
        init_values = [0, 0]
        final_values = [69, 31]
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

        # Animate through different iterations with changing sparsity
        start_iter = 1
        end_iter = self.total_number_of_iterations
        num_iterations = end_iter - start_iter
        
        def update_percent_symbols(bar: BarChart):
            # Position percent symbols next to their values
            symbolic_percent.next_to(symbolic_val, RIGHT, buff=0.05)
            numeric_percent.next_to(numeric_val, RIGHT, buff=0.05)

        def chart_updater(bar: BarChart):
            # Update the bar chart values and their positions
            sym_bar, num_bar = chart.bars
            chart.change_bar_values([symbolic_val.get_value(), numeric_val.get_value()])
            symbolic_val.next_to(sym_bar, UP, buff=0.1)
            numeric_val.next_to(num_bar, UP, buff=0.1)

        chart.add_updater(chart_updater, call_updater=True)
        symbolic_percent.add_updater(update_percent_symbols, call_updater=True)
        numeric_percent.add_updater(update_percent_symbols, call_updater=True)
        chart_brace = BraceLabel(chart, text=r"\text{Sparsity Analysis Overhead}", buff=0.1, font_size=22)
        
        # Animate everything
        self.scene.play(FadeIn(matrix_group), FadeIn(matrix_group_brace), FadeIn(box),
                FadeIn(chart), FadeIn(chart_brace), FadeIn(symbolic_percent), FadeIn(numeric_percent), run_time=self.FadeIn_time)
        self.scene.play(Create(box_to_matrix_arrow), runtime=self.arrow_time)

        # Animate matrix and chart updates for each iteration
        animation_time_per_iteration = self.per_solve_time
        for i, iteration in enumerate(range(start_iter, end_iter)):
            new_matrix_group = self.make_matrix_with_subtitle(matrix_font_size, iteration + 1, sparsity=0.1)
            new_matrix_group.move_to(matrix_group.get_center())
            new_box = SurroundingRectangle(application_block[iteration], color=RED, buff=0.1)
            new_arrow = Arrow(new_box.get_right(), new_matrix_group.get_left(), color=WHITE)
            new_matrix_group_brace = BraceLabel(new_matrix_group, text=r"\text{Sparse Cholesky Solve}", buff=0.1, font_size=22)
            new_transform_group = VGroup(new_box, new_arrow, new_matrix_group, new_matrix_group_brace)
            old_transform_group = VGroup(box, box_to_matrix_arrow, matrix_group, matrix_group_brace)

            # Animate both matrix transformation and gauge deceleration simultaneously
            if i != end_iter - 3:
                self.scene.play(
                    Transform(old_transform_group, new_transform_group),
                    ChangeDecimalToValue(symbolic_val, final_values[0] / num_iterations * (i + 1)),
                    ChangeDecimalToValue(numeric_val, final_values[1] / num_iterations * (i + 1)),
                    run_time=animation_time_per_iteration
                )
            else:
                # Highlight the symbolic inspector bar and show conclusion
                inspector_bar = chart.bars[0]
                inspector_bar.set_color(RED)
                inspector_bar.set_weight(BOLD)
                inspector_label = chart.x_axis.labels[0]
                conclusion = Tex(r"Sparsity analysis is \textcolor{red}{expensive}!", font_size=32)
                conclusion.to_edge(UP, buff=1)
                self.scene.play(
                    Transform(old_transform_group, new_transform_group),
                    ChangeDecimalToValue(symbolic_val, final_values[0] / num_iterations * (i + 1)),
                    ChangeDecimalToValue(numeric_val, final_values[1] / num_iterations * (i + 1)),
                    Write(conclusion),
                    run_time=animation_time_per_iteration
                )

        # Show a sad-face SVG above the symbolic bar to emphasize the bottleneck
        sad_face = SVGMobject("Figures/section1/sad-face.svg")
        sad_face.scale_to_fit_width(chart.bars[0].get_width())
        sad_face.next_to(symbolic_val, UP, buff=0.2)
        self.scene.play(FadeIn(sad_face), run_time=self.FadeIn_time)
        self.scene.wait(self.section1_planned_time - self.section1_total_time)

    def section1(self):
        # Entry point for Section 1 animation, with or without voiceover
        if isinstance(self.scene, VoiceoverScene):
            scripts = "Applications like second-order optimization require back-to-back sparse Cholesky solves, but dynamic sparsity imposes high analysis overhead!"
            with self.scene.voiceover(text=scripts) as tracker:    
                self.run_scene()
        else:
            self.run_scene()