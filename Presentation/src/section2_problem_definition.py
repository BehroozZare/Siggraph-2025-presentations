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
        print("HERE1", imgs)
        self.frame_list = [
            ImageMobject(f"scripts/obj_renders/images/{f}") for f in imgs
        ]

        # load & sort matrix renders
        mats = [f for f in os.listdir("scripts/matrix_vis/results/") if f.endswith(".png")]
        mats.sort(key=self._extract_frame_index)
        print("HERE2", mats)
        self.matrix_list = [
            ImageMobject(f"scripts/matrix_vis/results/{f}") for f in mats
        ]

        self.sim_frames = [self._show_simulation_frames(i) for i in range(len(self.frame_list))]

        assert len(self.frame_list) == len(self.matrix_list), \
            "Number of frames and matrices must match"
    

    def _show_simulation_frames(self, iteration: int)->Group:
        #Create a VGroup of the frame and the matrix
        frame = self.frame_list[iteration]
        matrix = self.matrix_list[iteration]
        frame.scale(1.2)
        matrix.scale_to_fit_width(frame.get_width())
        matrix.next_to(frame, LEFT, buff=1)
        #Adding a surronding box around the matrix
        matrix_box = SurroundingRectangle(matrix, buff=0.0, color=BLACK, stroke_width=1)
        return Group(frame, matrix, matrix_box)

    def _show_bottleneck_chart(self):
        pass


    def play_problem_definition(self):
        self._prepare_simulations_frames()
        # Entry point for Section 1 animation, with or without voiceover
        if isinstance(self.scene, VoiceoverScene):
            # script1 = "In practice, often time we are dealing with a sequence of Cholesky solves. Here, we can see a simplified second-order\
            #     Newton solver, where the direction of optimization should be computed in each iteration using Cholesky solves."
            # with self.scene.voiceover(text=script1) as tracker:
            #     algorithm_block = self._create_second_order_newton_solver().scale(0.8)
            #     algorithm_block.center()
            #     #Create a surronding box around the direction computation
            #     direction_computation_box = SurroundingRectangle(algorithm_block[3], buff=0.1, color=BLACK)
            #     self.scene.play(Create(algorithm_block), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)
            #     self.scene.play(Create(direction_computation_box), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)

            # self.scene.play(FadeOut(algorithm_block), FadeOut(direction_computation_box), run_time=self.transform_runtime)
            # script2 = "If the Hessian sparsity is constant across the iterations, the symbolic analysis can be performed once and reused for all iterations.\
            #     which leads to high-performance Cholesky solves!" # Show the matObBoard simulation
            
            # with self.scene.voiceover(text=script2) as tracker:    
            #     sparse_matrix = create_sparse_matrix(9, 0, 0.1)
            #     sparse_cholesky_framework = SymbolicNumericFramework(A_sp=sparse_matrix, matrix_size=9, generate_random_pattern=False,
            #                                                         generate_random_values=True, matrix_name="H", rhs_name="-g",
            #                                                         unknown_name="d")
            #     sparse_cholesky_framework.center()
            #     total_time = tracker.duration
            #     time_per_iteration = total_time / 15
            #     smile_emoji = ImageMobject("Figures/Problem/smile.png").scale(0.3)
            #     smile_emoji.next_to(sparse_cholesky_framework[0][0], UP, buff=1)
            #     self.scene.play(FadeIn(smile_emoji), run_time=self.transform_runtime)
            #     for i in range(15):
            #         numeric_color = RED_A if i % 2 == 0 else GREEN_A
            #         new_framework = SymbolicNumericFramework(A_sp=sparse_matrix, iteration=i, numeric_box_color=numeric_color,
            #                                                 matrix_size=9, generate_random_pattern=False, generate_random_values=True,
            #                                                 matrix_name="H", rhs_name="-g", unknown_name="d")
            #         new_framework.move_to(sparse_cholesky_framework.get_center())
            #         self.scene.play(Transform(sparse_cholesky_framework, new_framework), run_time=time_per_iteration)
                    


            # script3 = "However, if the sparsity pattern changes rapidely, the performance of Cholesky solves significantly decreases."
            # with self.scene.voiceover(text=script3) as tracker:
            #     sparse_cholesky_framework.center()
            #     total_time = tracker.duration
            #     time_per_iteration = total_time / 15
            #     sad_emoji = ImageMobject("Figures/Problem/crying.png").scale_to_fit_width(smile_emoji.get_width())
            #     sad_emoji.move_to(smile_emoji.get_center())
            #     self.scene.play(ReplacementTransform(smile_emoji, sad_emoji), run_time=self.transform_runtime)
            #     for i in range(15):
            #         numeric_color = RED_A if i % 2 == 0 else GREEN_A
            #         symbolic_color = RED_A if i % 2 == 0 else YELLOW_B
            #         new_framework = SymbolicNumericFramework(A_sp=sparse_matrix, iteration=i, numeric_box_color=numeric_color, symbolic_box_color=symbolic_color,
            #                                                 matrix_size=9, generate_random_pattern=True, generate_random_values=True,
            #                                                 matrix_name="H", rhs_name="-g", unknown_name="d")
            #         new_framework.move_to(sparse_cholesky_framework.get_center())
            #         self.scene.play(Transform(sparse_cholesky_framework, new_framework), run_time=time_per_iteration)

            # Show an example with and withot permutation vector
            script4 = "This is actually a scenario in applications such as well-known incremental potential contact framework.\
                here, we can see the simualtion of a mat which falls on a board and snapshots of hessians sparsity during the simulation.\
                As can be seen, up to 70\% of runtime is spend on symbolic analysis for this simulation."
            with self.scene.voiceover(text=script4) as tracker:
                # self.scene.play(FadeOut(sparse_cholesky_framework), FadeOut(sad_emoji), run_time=self.transform_runtime)
                frame_and_matrix = self.sim_frames[0]
                frame_and_matrix.center()
                self.scene.play(FadeIn(frame_and_matrix), run_time=self.transform_runtime)
                total_time = tracker.duration
                time_per_iteration = total_time / (len(self.frame_list) - 1)
                rt = max(0.2, time_per_iteration)
                for i in range(1, len(self.frame_list)):
                    new_frame_and_matrix = self.sim_frames[i]
                    new_frame_and_matrix.move_to(frame_and_matrix.get_center())
                    self.scene.remove(frame_and_matrix)
                    self.scene.add(new_frame_and_matrix)
                    self.scene.wait(0.1)
                    frame_and_matrix = new_frame_and_matrix
                
        else:
            pass