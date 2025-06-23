from manim import *
import numpy as np


# 2) extend the pre-amble
template = TexTemplate()
template.add_to_preamble(r"\usepackage{xcolor}")

config.tex_template = template
# xelatex_tpl.add_to_preamble(r"\usepackage{fontspec}")
# xelatex_tpl.add_to_preamble(r"\setmainfont{Linux Biolinum}")


class SpeedGauge(Group):
    def __init__(self, radius=1.0, min_value=0, max_value=100,
                position=ORIGIN, stroke_width=16, **kwargs):
        super().__init__(**kwargs)
        self.position = position
        self.radius = radius
        self.min_value = min_value
        self.max_value = max_value
        self.current_value = min_value
        self.stroke_width = stroke_width

        # Create the value arc (starts at 0)
        self.value_arc = Arc(
            radius=radius,
            start_angle=0,
            angle=0,  # Start with no arc
            color=RED,
            stroke_width=self.stroke_width,
            arc_center=self.position,
        )
        
        # Create the number display
        self.number_display = DecimalNumber(
            self.current_value,
            num_decimal_places=0,
            font_size=36,
            color=WHITE
        ).move_to(self.position)

        # Add all components to the group
        self.add(self.value_arc, self.number_display)
    
    def _value_to_angle(self, value):
        """Convert a value to an angle between 0 and PI"""
        if self.max_value == self.min_value:
            return 0
        normalized = (value - self.min_value) / (self.max_value - self.min_value)
        return normalized * PI
    
    def _value_to_color(self, value):
        """Convert a value to a color between RED and BLUE"""
        if self.max_value == self.min_value:
            return RED
        normalized = (value - self.min_value) / (self.max_value - self.min_value)
        return interpolate_color(RED, BLUE, normalized)
    
    def create_initial_animation(self, scene, value=None, run_time=1.0):
        """Create the initial animation to show the gauge"""
        if value is None:
            value = self.min_value
        
        # Set initial value without animation
        self.current_value = value
        angle = self._value_to_angle(value)
        color = self._value_to_color(value)
        
        self.value_arc.set_color(color)
        self.value_arc.set_stroke_width(self.stroke_width)
        self.value_arc.set_angle(angle)
        self.value_arc.set_arc_center(self.position)
    
        self.number_display.set_value(value)
        
        # Return an animation that creates the arc and shows the number
        return AnimationGroup(
            Create(self.value_arc),
            Write(self.number_display, run_time=0),
            run_time=run_time
        )
    
    def set_value(self, new_value, run_time=1.0):
        """Animate the gauge to a new value"""
        # Clamp the value to the valid range
        new_value = max(self.min_value, min(self.max_value, new_value))
        
        new_angle = self._value_to_angle(new_value)
        new_color = self._value_to_color(new_value)
        
        old_angle = self.value_arc.angle
        old_color = self.value_arc.color
        # Define updater: rebuild arc per-frame
        def update_arc(arc, alpha):
            t = linear(alpha)
            angle = old_angle + (new_angle - old_angle) * t
            color = interpolate_color(old_color, new_color, t)
            arc.become(Arc(
                radius=self.radius,
                start_angle=0,
                angle=angle,
                color=color,
                stroke_width=self.stroke_width,
                arc_center=self.position,
            ))
            arc.set_color(color)
            arc.set_angle(angle)
            arc.set_stroke_width(self.stroke_width)
        
        animation = AnimationGroup(
            UpdateFromAlphaFunc(self.value_arc, update_arc, run_time=run_time),
            ChangeDecimalToValue(self.number_display, new_value),
            run_time=run_time
        )
    
        
        return animation



        
class ProblemDefinition(Scene):
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

    def construct(self):
        # # Make the title and start it in the center
        # title = self.make_title()
        # title.to_edge(UP)
        # self.add(title)

        # # Show title in center first
        # self.play(FadeIn(title))
        
        # # Transition title to top with reduced font size
        # self.play(
        #     title.animate.to_edge(UP).scale(0.6),
        #     run_time=0.5
        # )
        # self.wait(0.5)
        
        # Start with what is important for us
        explanation = Tex(r"Performing a \textcolor{red}{sequence} of \textcolor{green}{Sparse Cholesky solves} is common!", 
                          font_size=36, color=BLUE)
        explanation.to_edge(UP, buff=1)
        self.play(Write(explanation))

        self.wait(1)

        # Using the algorithm block to show an example of a sequence of linear systems
        algorithm_block = self.make_algorithm_block()
        algorithm_block.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        
        # Position the algorithm block below the explanation
        algorithm_block.next_to(explanation, DOWN, buff=2)
        algorithm_block.to_edge(LEFT)
        self.add(algorithm_block)
        self.wait(0.5)


        # Add a surrounding box around the algorithm_block[3]
        box = SurroundingRectangle(algorithm_block[3], color=RED, buff=0.1)
        self.play(Create(box))
        self.wait(0.5)
        
        # Create the matrix with subtitle (iteration 1)
        matrix_font_size = 32
        matrix_group = self.make_matrix_with_subtitle(matrix_font_size, 1, sparsity=0.1)
        matrix_group.next_to(algorithm_block, RIGHT, buff=0.5)
        # matrix_group.move_to(ORIGIN)
        
        # Create arrow from the Hd=-g line to the matrix
        arrow = Arrow(
            algorithm_block[3].get_right() + RIGHT * 0.1,
            matrix_group.get_left() + LEFT * 0.2,
            color=YELLOW,
            buff=0.1
        )

        # First draw the arrow
        self.play(Create(arrow), run_time=0.1)
        
        # Show the matrix
        self.play(FadeIn(matrix_group), runtime=0.5)
        
        init_values = [0, 0]
        final_values = [80, 20]
        # Create the bar chart that shows the inspector bottleneck
        chart = BarChart(
            values=init_values,
            bar_names=["Inspector", "Numerics"],
            y_range=[0, 100, 20],
            y_length=2,
            x_length=3,
            x_axis_config={
                "font_size": 22,
                "label_constructor": lambda text: Tex(text, font_size=22).shift(DOWN * 0.5)  # Move labels down by 0.3 units
            },
            y_axis_config={"font_size": 22}
        )
        chart.next_to(matrix_group, RIGHT, buff=0.2)
        
        # Manually rotate the x-axis labels by 45 degrees
        for label in chart.x_axis.labels:
            label.rotate(PI/4)  # 45 degrees rotation

        
        # Animate through different iterations with changing sparsity
        # Calculate the deceleration values for the gauge
        start_iter = 2
        end_iter = 8
        num_iterations = end_iter - start_iter
        
        for i, iteration in enumerate(range(start_iter, end_iter)):
            temperoray_final_values = [final_values[0] / num_iterations * (i + 1), final_values[1] / num_iterations * (i + 1)]

            # Create new matrix with different sparsity
            new_matrix_group = self.make_matrix_with_subtitle(matrix_font_size, iteration, sparsity=0.1)
            new_matrix_group.move_to(matrix_group.get_center())

            # Animate both matrix transformation and gauge deceleration simultaneously
            self.play(
                Transform(matrix_group, new_matrix_group),
                chart.animate.change_bar_values(temperoray_final_values),
                run_time=0.5
            )

        
        #Making the inspector color in bar chart bold and moving
        inspector_bar = chart.bars[0]
        inspector_bar.set_color(RED)
        inspector_bar.set_weight(BOLD)
        self.play(inspector_bar.animate.shift(UP * 0.1), run_time=0.2)
        self.play(inspector_bar.animate.shift(DOWN * 0.1), run_time=0.2)
        # Make the label bold
        inspector_label = chart.x_axis.labels[0]
        original_color = inspector_label.color
        self.play(
            inspector_label.animate.set_color(YELLOW).shift(DOWN * 0.1),
            run_time=0.2
        )
        self.play(
            inspector_label.animate.set_color(original_color).shift(UP * 0.1),
            run_time=0.2
        )
        


        # Start with what is important for us
        problem_sentence = Tex(r"However, it is not fast when the \textcolor{red}{sparsity pattern changes} rapidely!", 
                          font_size=36)
        problem_sentence.next_to(explanation, DOWN, buff=0.2)
        self.play(Write(problem_sentence))
        
        