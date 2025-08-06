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
        hessian_label = BraceLabel(matrix_box, text="Hessians", label_constructor=Tex, buff=0.1, font_size=32).set_color(BLACK)
        mat_on_board_label = BraceLabel(frame, text="IPC:MatOnBoard", label_constructor=Tex, buff=0.1, font_size=32).set_color(BLACK)
        return Group(frame, matrix, matrix_box, hessian_label, mat_on_board_label).scale_to_fit_height(6)

    def _create_bar_chart(self, symbol: str = "\%")->CustomBarChart:
        init_vals   = [0.1, 0.1, 0.1]
        names       = ["MKL Pardiso", "Accelerate", "CHOLMOD"]

        chart = CustomBarChart(
            init_vals,                 # initial bar heights
            bar_names=names,
            y_range=[0, 100, 20],
            y_length=2,
            x_length=3,
            label_font_size=22,
            scale_symbol=symbol,
        )
        return chart
    
    def _create_paper_initial_sparse_matrix(self) -> np.ndarray:
        # Create the COO list
        row_indices = [0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8]
        col_indices = [0, 0, 1, 1, 2, 0, 2, 3, 0, 4, 4, 5, 5, 6, 1, 6, 7, 2, 4, 8]
        #Define random values in the size of row_indices or col_indices
        values = np.random.randn(len(row_indices))
        # sum each row values and add the sum value to the diagonal value
        row_sum_values = np.zeros(9)
        for i in range(len(row_indices)):
            if row_indices[i] != col_indices[i]:
                row_sum_values[row_indices[i]] += values[i]
        
        #Create the dense ndarray matrix which 9x9
        dense_matrix = np.zeros((9, 9))
        for i in range(len(row_indices)):
            dense_matrix[row_indices[i], col_indices[i]] = values[i]
            dense_matrix[col_indices[i], row_indices[i]] = values[i]

        #Sum the values in each row (Without the diagonal) to make it SPD
        for i in range(len(row_indices)):
            dense_matrix[row_indices[i], row_indices[i]] += row_sum_values[row_indices[i]]

        return dense_matrix
    
    def _global_permutation_vector(self, text: str = "$P_G:$", colorful: bool = False)->VGroup:
        color_nodes={0: PURPLE, 1: GREEN, 2: YELLOW, 3: BLUE, 4: GOLD, 5: MAROON, 6: TEAL}
        post_order_idx = [3, 4, 1, 5, 6, 2, 0]
        global_perm_values = [7, 5, 6, 3, 8, 2, 1, 4, 0]
        transpose = []
        for i in range(len(global_perm_values)):
            transpose.append([global_perm_values[i]])
        global_permutation_vector = Matrix([transpose])
        global_permutation_vector.get_brackets().set_color(BLACK)
        if colorful:
            for i in range(len(global_permutation_vector.get_entries())):
                if i < 6:
                    global_permutation_vector.get_entries()[i].set_color(color_nodes[post_order_idx[i]])
                else:
                    global_permutation_vector.get_entries()[i].set_color(color_nodes[post_order_idx[6]])
        else:
            global_permutation_vector.get_entries().set_color(BLACK)
        label = Tex(text, color=BLACK)
        global_permutation_vector = VGroup(label, global_permutation_vector).arrange(DOWN, buff=0.2)
        return global_permutation_vector

    def play_problem_definition(self):
        self._prepare_simulations_frames()
        # Entry point for Section 1 animation, with or without voiceover
        if isinstance(self.scene, VoiceoverScene):
            script1 = "Unfortunately, not all applications are friendly enough to have constant sparsity pattern."
            with self.scene.voiceover(text=script1) as tracker:
                pass

            script2 = "For example, here we have an application of physics-based simulation involving contact using Incremental Potential Contact or (IPC) simulator."
            with self.scene.voiceover(text=script2) as tracker:
                frame_and_matrix = self.sim_frames[0]
                frame_and_matrix.center()
                self.scene.play(FadeIn(frame_and_matrix), run_time=self.transform_runtime)
                total_time = tracker.duration
                time_per_iteration = total_time / (len(self.frame_list) - 1)
                rt = max(0.2, time_per_iteration)
                for i in range(1, len(self.sim_frames)):
                # for i in range(1, 4):
                    new_frame_and_matrix = self.sim_frames[i]
                    new_frame_and_matrix.center()
                    self.scene.remove(frame_and_matrix)
                    self.scene.add(new_frame_and_matrix)
                    self.scene.wait(0.1)
                    frame_and_matrix = new_frame_and_matrix

            script3 = "In this application, multiple iterations of a second-order optimizer are performed for each frame.\
                In every iteration, a direct linear solver is used. Due to the mechanics of IPC for preventing contact,\
                non-zero entries are added to enforce contact prevention."
            with self.scene.voiceover(text=script3) as tracker:
                pass

            script4 = "The computational pattern for these kind of simulation is like this,\
                where both symbolic and numerical phase are called rapidely, leading to expensive symbolic analysis overhead."
            with self.scene.voiceover(text=script4) as tracker:
                self.scene.play(FadeOut(frame_and_matrix), run_time=self.transform_runtime)
                sparse_matrix = create_sparse_matrix(9, 0, 0.1)
                sparse_cholesky_solver = SymbolicNumericFramework(A_sp=sparse_matrix, matrix_size=9,
                                                                generate_random_pattern=True, generate_random_values=True,
                                                                matrix_name="H", rhs_name="-g", unknown_name="d")
                sparse_cholesky_solver.center()
                total_time = tracker.duration
                time_per_iteration = total_time / 10
                sad_emoji = ImageMobject("Figures/Problem/crying.png").scale(0.3)
                sad_emoji.next_to(sparse_cholesky_solver[0][0], UP, buff=1)
                for i in range(10):
                    numeric_color = RED_A if i % 2 == 0 else GREEN_A
                    symbolic_color = RED_A if i % 2 == 0 else YELLOW_B
                    new_framework = SymbolicNumericFramework(A_sp=sparse_matrix, iteration=i, numeric_box_color=numeric_color, symbolic_box_color=symbolic_color,
                                                            matrix_size=9, generate_random_pattern=True, generate_random_values=True,
                                                            matrix_name="H", rhs_name="-g", unknown_name="d")
                    new_framework.move_to(sparse_cholesky_solver.get_center())
                    if i == 5:
                        self.scene.play(Transform(sparse_cholesky_solver, new_framework), FadeIn(sad_emoji), run_time=time_per_iteration)
                    else:
                        self.scene.play(Transform(sparse_cholesky_solver, new_framework), run_time=time_per_iteration)
            self.scene.play(FadeOut(sad_emoji, sparse_cholesky_solver), run_time=self.transform_runtime)

            script5 = "Our benchmark for these type application show that on average 67% of the Cholesky solver runtime is spent on the symbolic analysis phase."
            with self.scene.voiceover(text=script5) as tracker:
                frame_and_matrix = self.sim_frames[0].scale_to_fit_width(6)
                frame_and_matrix.move_to(-3.5 * RIGHT)
                self.scene.play(FadeIn(frame_and_matrix), run_time=self.transform_runtime)
                total_time = tracker.duration
                time_per_iteration = total_time / (len(self.frame_list) - 1)
                rt = max(0.2, time_per_iteration)
                for i in range(1, len(self.sim_frames)):
                # for i in range(1, 4):
                    new_frame_and_matrix = self.sim_frames[i].scale_to_fit_width(6)
                    new_frame_and_matrix.move_to(-3.5 * RIGHT)
                    self.scene.remove(frame_and_matrix)
                    self.scene.add(new_frame_and_matrix)
                    self.scene.wait(0.1)
                    frame_and_matrix = new_frame_and_matrix

                chart = self._create_bar_chart()
                chart.move_to(3.5 * RIGHT)
                chart.scale_to_fit_width(5)
                text = Text("Symbolic Analysis Overhead", font_size=32, color=BLACK).next_to(chart, UP, buff=0.5)
                self.scene.play(FadeIn(chart), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)
                self.scene.play(chart.animate_to_values([76,70,49], run_time=1))
                self.scene.play(Write(text), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)
                
            script6 = "To further investigate this problem, we also evaluated the internals of symbolic analysis,\
                where we can simplify as a fill-reducing ordering step plus rest of symbolic analysis overhead such as supernodal computation."
            with self.scene.voiceover(text=script6) as tracker:
                self.scene.play(FadeOut(frame_and_matrix, chart, text), run_time=self.transform_runtime)
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

            script7 = "Our internal benchmarking indicates that the fill-reducing ordering step is the primary bottleneck.\
                For example, in the MatOnBoard simulation, this step accounts for up to 85% of the symbolic analysis runtime."
            with self.scene.voiceover(text=script7) as tracker:
                self.scene.play(detailed_symbolic.animate.scale_to_fit_width(5).move_to(-3.5 * RIGHT), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

                #Create a bar chart for the symbolic analysis components
                ordering_chart = self._create_bar_chart()
                ordering_chart.move_to(3.5 * RIGHT)
                ordering_chart.scale_to_fit_width(5)
                ordering_text = Text("Fill-reducing Ordering overhead", font_size=32, color=BLACK).next_to(ordering_chart, UP, buff=0.5)
                self.scene.play(FadeIn(ordering_chart), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)
                self.scene.play(ordering_chart.animate_to_values([62,85,81], run_time=1))
                self.scene.play(Write(ordering_text), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

            script9 = "The objective of this module is to provide a permutation vector that reorders\
                the input matrix to reduce fill-ins during factorization. Specifically, the module takes\
                the sparsity pattern of matrix A as input and returns a permutation vector P such that the reordered matrix\
                has fewer fill-ins during factorization."
            with self.scene.voiceover(text=script9) as tracker:
                self.scene.play(FadeOut(detailed_symbolic[1], detailed_symbolic[2], ordering_chart, ordering_text), run_time=self.transform_runtime)
                fill_reducing_box = detailed_symbolic[0]
                fill_reducing_box.scale_to_fit_width(3)
                initial_sparse_matrix = self._create_paper_initial_sparse_matrix()
                initial_sparse_matrix_tex = create_manim_Matrix(row_num=initial_sparse_matrix.shape[0],
                                                                col_num=initial_sparse_matrix.shape[1],
                                                                matrix=initial_sparse_matrix)
                initial_sparse_matrix_tex.get_brackets().set_color(BLACK)
                initial_sparse_matrix_tex.scale_to_fit_width(4)
                initial_sparse_matrix_tex.next_to(fill_reducing_box, LEFT, buff=1)
                output_vector = np.array([7, 5, 6, 3, 8, 2, 1, 4, 0]).reshape(-1, 1)
                output_vector_tex = Matrix(output_vector)
                output_vector_tex.next_to(fill_reducing_box, RIGHT, buff=1)
                output_vector_tex.scale_to_fit_height(initial_sparse_matrix_tex.get_height())
                output_vector_tex.get_brackets().set_color(BLACK)
                output_vector_tex.get_entries().set_color(BLACK)

                #Arrows
                input_to_order = Arrow(initial_sparse_matrix_tex.get_right(), fill_reducing_box.get_left(), buff=0.1, color=BLACK)
                order_to_output = Arrow(fill_reducing_box.get_right(), output_vector_tex.get_left(), buff=0.1, color=BLACK)

                example_group = VGroup(initial_sparse_matrix_tex, fill_reducing_box, output_vector_tex, input_to_order, order_to_output)
                example_group.center()
                self.scene.play(Transform(fill_reducing_box, example_group), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)


            script10 = "Removing this step is not possible, as for example, here the factor memory footprint is exploded! As a result,\
                we need fast fill-reducing ordering algorithm for dynamic sparsity pattern!"
            with self.scene.voiceover(text=script10) as tracker:
                #load 3 images
                self.scene.play(FadeOut(example_group), run_time=self.transform_runtime)
                hessian_0_0_last_IPC_spy = ImageMobject("Figures/Problem/hessian_139_0_last_IPC_spy.png")
                hessian_0_0_last_IPC_cholesky_natural_ordering = ImageMobject("Figures/Problem/hessian_139_0_last_IPC_cholesky_natural_ordering.png")
                hessian_0_0_last_IPC_cholesky_metis_ordering = ImageMobject("Figures/Problem/hessian_139_0_last_IPC_cholesky_metis_ordering.png")
                matrices = Group(hessian_0_0_last_IPC_spy, hessian_0_0_last_IPC_cholesky_natural_ordering, hessian_0_0_last_IPC_cholesky_metis_ordering).arrange(RIGHT, buff=2)
                matrices.scale_to_fit_width(13)
                org_brace_label = BraceLabel(hessian_0_0_last_IPC_spy, text="Original (99.84\%)", label_constructor=Tex, buff=0.1, font_size=32).set_color(BLACK)
                natural_brace_label = BraceLabel(hessian_0_0_last_IPC_cholesky_natural_ordering, text="No Ordering (82.24\%)", label_constructor=Tex, buff=0.1, font_size=32).set_color(BLACK)
                metis_brace_label = BraceLabel(hessian_0_0_last_IPC_cholesky_metis_ordering, text="Metis Ordering (99.20\%)", label_constructor=Tex, buff=0.1, font_size=32).set_color(BLACK)
                self.scene.play(FadeIn(matrices), run_time=self.transform_runtime)
                self.scene.play(FadeIn(org_brace_label), FadeIn(natural_brace_label), FadeIn(metis_brace_label), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)