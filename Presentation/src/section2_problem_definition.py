from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
from utils import *
import numpy as np
import os, re

# 2) extend the pre-amble
template = TexTemplate()
template.add_to_preamble(r"\usepackage{xcolor}")

config.tex_template = template


# Section1: Handles the animation and logic for the problem definition section of the FastTrack video.
class ProblemDefinition():
    def __init__(self, scene: Scene | VoiceoverScene):
        # Initialize the section with timing and animation parameters
        self.scene = scene
        self.transform_runtime = 0.5
        self.wait_time = 1
        self.frame_list = []
        self.matrix_list = []
        self.sim_frames = []

    def _extract_frame_index(self, filename: str) -> int:
        m = re.search(r"(\d+)", filename)
        if not m:
            raise ValueError(f"No digits in {filename!r}")
        return int(m.group(1))
    
    def _create_second_order_newton_solver(self) -> VGroup:
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
            lines.append(Tex(line, tex_template=template, font_size=32, color=BLACK))
            
        algorithm_block = VGroup(*lines)
        algorithm_block.arrange(DOWN, aligned_edge=LEFT, buff=0.3)

        return algorithm_block

    def _prepare_simulations_frames(self):
        # load & sort object renders
        imgs = [f for f in os.listdir("scripts/obj_renders/images/") if f.endswith(".png")]
        imgs.sort(key=self._extract_frame_index)
        self.frame_list = [
            ImageMobject(f"scripts/obj_renders/images/{f}") for f in imgs
        ]

        # load & sort matrix renders
        mats = [f for f in os.listdir("scripts/matrix_vis/results/") if f.endswith(".png")]
        mats.sort(key=self._extract_frame_index)
        self.matrix_list = [
            ImageMobject(f"scripts/matrix_vis/results/{f}") for f in mats
        ]
        min_size = min(len(self.frame_list), len(self.matrix_list))

        self.sim_frames = [self._show_simulation_frames(i) for i in range(min_size)]
    

    def _show_simulation_frames(self, iteration: int)->Group:
        #Create a VGroup of the frame and the matrix
        frame = self.frame_list[iteration]
        matrix = self.matrix_list[iteration]
        frame.scale(1.2)
        matrix.scale_to_fit_width(frame.get_width())
        matrix.next_to(frame, LEFT, buff=1)
        #Adding a surronding box around the matrix
        matrix_box = SurroundingRectangle(matrix, buff=0.0, color=BLACK, stroke_width=1)
        hessian_label = BraceLabel(matrix_box, text=r"Hessians", buff=0.1, font_size=32).set_color(BLACK)
        mat_on_board_label = BraceLabel(frame, text=r"IPC:MatOnBoard", buff=0.1, font_size=32).set_color(BLACK)
        return Group(frame, matrix, matrix_box, hessian_label, mat_on_board_label)

    def _create_bar_chart(self)->CustomBarChart:
        init_vals   = [0.1, 0.1, 0.1]
        names       = ["MKL Pardiso", "Accelerate", "CHOLMOD"]

        chart = CustomBarChart(
            init_vals,                 # initial bar heights
            bar_names=names,
            y_range=[0, 100, 20],
            y_length=2,
            x_length=3,
            label_font_size=22,
        )
        return chart


    def play_problem_definition(self):
        self._prepare_simulations_frames()
        # Entry point for Section 1 animation, with or without voiceover
        if isinstance(self.scene, VoiceoverScene):
            script1 = "The two-step framework allows symbolic reuse when the sparsity pattern remains constant and only\
                the numerical values change. It’s useful in scenarios requiring a sequence of Cholesky solves.\
                Here, we illustrate a simplified second-order Newton solver, where computing the optimization direction naturally forms such a sequence."
            with self.scene.voiceover(text=script1) as tracker:
                algorithm_block = self._create_second_order_newton_solver().scale(0.8)
                algorithm_block.center()
                #Create a surronding box around the direction computation
                direction_computation_box = SurroundingRectangle(algorithm_block[3], buff=0.1, color=BLACK)
                self.scene.play(Create(algorithm_block), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)
                self.scene.play(Create(direction_computation_box), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

            self.scene.play(FadeOut(algorithm_block), FadeOut(direction_computation_box), run_time=self.transform_runtime)
            script2 = "Here, we can see this framework in action, when the sparsity pattern is constant and only the numerical values change." # Show the matObBoard simulation
            
            with self.scene.voiceover(text=script2) as tracker:    
                sparse_matrix = create_sparse_matrix(9, 0, 0.1)
                sparse_cholesky_framework = SymbolicNumericFramework(A_sp=sparse_matrix, matrix_size=9, generate_random_pattern=False,
                                                                    generate_random_values=True, matrix_name="H", rhs_name="-g",
                                                                    unknown_name="d")
                sparse_cholesky_framework.center()
                total_time = tracker.duration
                time_per_iteration = total_time / 10
                smile_emoji = ImageMobject("Figures/Problem/smile.png").scale(0.3)
                smile_emoji.next_to(sparse_cholesky_framework[0][0], UP, buff=1)
                self.scene.play(FadeIn(smile_emoji), run_time=self.transform_runtime)
                for i in range(10):
                    numeric_color = RED_A if i % 2 == 0 else GREEN_A
                    new_framework = SymbolicNumericFramework(A_sp=sparse_matrix, iteration=i, numeric_box_color=numeric_color,
                                                            matrix_size=9, generate_random_pattern=False, generate_random_values=True,
                                                            matrix_name="H", rhs_name="-g", unknown_name="d")
                    new_framework.move_to(sparse_cholesky_framework.get_center())
                    self.scene.play(Transform(sparse_cholesky_framework, new_framework), run_time=time_per_iteration)
                    


            script3 = "However, if the sparsity pattern changes rapidely, the performance of Cholesky solves \
                significantly decreases, as it requires recomputation of expensive symbolic analysis."
            with self.scene.voiceover(text=script3) as tracker:
                sparse_cholesky_framework.center()
                total_time = tracker.duration
                time_per_iteration = total_time / 10
                sad_emoji = ImageMobject("Figures/Problem/crying.png").scale_to_fit_width(smile_emoji.get_width())
                sad_emoji.move_to(smile_emoji.get_center())
                self.scene.play(ReplacementTransform(smile_emoji, sad_emoji), run_time=self.transform_runtime)
                for i in range(10):
                    numeric_color = RED_A if i % 2 == 0 else GREEN_A
                    symbolic_color = RED_A if i % 2 == 0 else YELLOW_B
                    new_framework = SymbolicNumericFramework(A_sp=sparse_matrix, iteration=i, numeric_box_color=numeric_color, symbolic_box_color=symbolic_color,
                                                            matrix_size=9, generate_random_pattern=True, generate_random_values=True,
                                                            matrix_name="H", rhs_name="-g", unknown_name="d")
                    new_framework.move_to(sparse_cholesky_framework.get_center())
                    self.scene.play(Transform(sparse_cholesky_framework, new_framework), run_time=time_per_iteration)
            self.scene.wait(self.wait_time)
            self.scene.play(FadeOut(sparse_cholesky_framework), FadeOut(sad_emoji), run_time=self.transform_runtime)

            # Show an example with and withot permutation vector
            script4 = "Applications such as well-known incremental potential contact framework fall into this computational pattern.\
                here, hessians and corresponding frames are shown. As can be seen the sparsity pattern changes rapidly across calls to sparse cholesky solve."
            with self.scene.voiceover(text=script4) as tracker:
                frame_and_matrix = self.sim_frames[0]
                frame_and_matrix.center()
                self.scene.play(FadeIn(frame_and_matrix), run_time=self.transform_runtime)
                total_time = tracker.duration
                time_per_iteration = total_time / (len(self.frame_list) - 1)
                rt = max(0.2, time_per_iteration)
                for i in range(1, len(self.sim_frames)):
                # for i in range(1, 4):
                    new_frame_and_matrix = self.sim_frames[i]
                    new_frame_and_matrix.move_to(frame_and_matrix.get_center())
                    self.scene.remove(frame_and_matrix)
                    self.scene.add(new_frame_and_matrix)
                    self.scene.wait(0.1)
                    frame_and_matrix = new_frame_and_matrix

            # Add the chart for symbolic
            script5 = "Our analysis indicates that up to 76\% of runtime is spend on symbolic analysis for this simulation."
            with self.scene.voiceover(text=script5) as tracker:
                self.scene.play(frame_and_matrix.animate.to_edge(LEFT, buff=1), run_time=self.transform_runtime)
                chart = self._create_bar_chart()
                chart.to_edge(RIGHT, buff=1)
                self.scene.play(FadeIn(chart), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)
                self.scene.play(chart.animate_to_values([76,70,49], run_time=1))
                self.scene.wait(self.wait_time)
            

            script6 = "Our detailed benchmark of symbolic analysis components further indicates \
                that the bulk of the symbolic analysis runtime is spent on fill-reducing ordering computation."
            with self.scene.voiceover(text=script6) as tracker:
                self.scene.play(FadeOut(frame_and_matrix), FadeOut(chart), run_time=self.transform_runtime)
                symbolic_box = moduleBox(label_text="Symbolic Analysis", font_size=32, text_color=BLACK, stroke_color=BLACK,
                                          block_total_width=4, block_total_height=2.0, fill_color=YELLOW_A, corner_radius=0.1)
                symbolic_box.center()
                self.scene.play(FadeIn(symbolic_box), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

                #Create a bar chart for the symbolic analysis components
                ordering_box = moduleBox(label_text="Fill-reducing Ordering", font_size=32, text_color=BLACK, stroke_color=BLACK,
                                          block_total_width=4.0, block_total_height=2.0, fill_color=YELLOW_A, corner_radius=0.1)
                rest_of_analysis_box = moduleBox(label_text="Rest of Analysis", font_size=32, text_color=BLACK, stroke_color=BLACK,
                                                block_total_width=3.0, block_total_height=2.0, fill_color=YELLOW_A, corner_radius=0.1)
                rest_of_analysis_box.next_to(ordering_box, RIGHT, buff=1)
                order_to_rest_arrow = Arrow(ordering_box.get_right(), rest_of_analysis_box.get_left(), buff=0.1, color=BLACK)
                detailed_symbolic = VGroup(ordering_box, rest_of_analysis_box, order_to_rest_arrow)
                detailed_symbolic.move_to(symbolic_box.get_center())
                self.scene.play(ReplacementTransform(symbolic_box, detailed_symbolic), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

            script7 = "Here, for Mat On Board simulation, fill-reducing ordering computation takes up to 85\% of the symbolic analysis runtime."
            with self.scene.voiceover(text=script7) as tracker:
                self.scene.play(detailed_symbolic.animate.to_edge(LEFT, buff=1), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

                #Create a bar chart for the symbolic analysis components
                ordering_chart = self._create_bar_chart()
                ordering_chart.to_edge(RIGHT, buff=1)
                self.scene.play(FadeIn(ordering_chart), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)
                self.scene.play(ordering_chart.animate_to_values([62,85,81], run_time=1))
                self.scene.wait(self.wait_time)

            script8 = "Fill-reducing ordering involves finding a permutation that minimizes the fill-in of the Cholesky factorization,\
                and it is a crucial step for fast sparse cholesky solve."
            with self.scene.voiceover(text=script8) as tracker:
                self.scene.play(FadeOut(ordering_chart), run_time=self.transform_runtime)
                self.scene.play(detailed_symbolic.animate.center(), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)
                #PAP^T equation
                pap_t_equation = Tex(r"$PAP^T$", font_size=32, color=BLACK)
                pap_t_equation.next_to(detailed_symbolic[2], LEFT, buff=0.1)
                self.scene.play(Transform(detailed_symbolic[0], pap_t_equation), run_time=self.transform_runtime)
                self.scene.play(detailed_symbolic.animate.center(), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

        else:
            pass