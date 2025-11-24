from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import cholesky
from utils import *
import os
import re

from slides.SCENE_CONFIG import *

# 2) extend the pre-amble
template = TexTemplate()
template.add_to_preamble(r"\usepackage{xcolor}")
template.add_to_preamble(r"\usepackage{helvet}")  # For Helvetica font
template.add_to_preamble(r"\renewcommand{\familydefault}{\sfdefault}")  # Set sans-serif as default

config.tex_template = template


# Section1: Handles the animation and logic for the problem definition section of the FastTrack video.
class Background():
    def __init__(self, scene: Scene | VoiceoverScene):
        # Initialize the section with timing and animation parameters
        self.scene = scene
        self.size = 9
        self.text_color = TEXT_COLOR
        self.scale_factor = 0.6
        self.label_font_size = FONT_SIZE
        self.value_font_size = 18
        self.transform_runtime = 0.5
        self.forward_backward_creation_runtime = 2
        self.arrow_runtime = 0.5
        self.arrow_stroke_width = 1
        self.wait_time = 1

        self.dense_matrix = None
        self.dense_lower_triangular_matrix = None
        self.dense_upper_triangular_matrix = None
        self.sparse_matrix = None
        self.sparse_lower_triangular_matrix = None
        self.sparse_upper_triangular_matrix = None
        
        
        #Solver pipeline variables
        self.linear_sys_definition = None
        self.dense_eq = None
        self.dense_llt_eq = None
        self.dense_forward_eq = None
        self.dense_backward_eq = None
        self.dense_forward_backward_eq = None
        self.dense_forward_backward_brace = None    
        self.sparse_eq = None
        self.sparse_llt_eq = None
        self.sparse_forward_eq = None
        self.sparse_backward_eq = None
        self.sparse_forward_backward_eq = None
        self.sparse_forward_backward_brace = None


        self.sim_frames = []

    def set_current_scene(self, scene: Scene | VoiceoverScene):
        self.scene = scene


    def cholesky_sparsity_pattern(self, A_dense: np.ndarray, ordering_method: str = 'metis') -> np.ndarray:
        # Use scipy.linalg.cholesky for dense matrices
        #Make the A_dense SPD by adding a small positive value to the diagonal
        A_dense = A_dense + 100 * np.eye(A_dense.shape[0])

        L = cholesky(A_dense, lower=True)
        # Convert to binary pattern
        L_pattern = (L != 0).astype(np.int8)
        return L_pattern


    def _get_centers_of_section(self, num_sections: int) -> list[Dot]:
        # 1) get full frame width
        W = self.scene.camera.frame_width
        # 2) compute each column's width
        w = W / num_sections
        # 3) build a list of the 3 mid‐points
        centers = [
            np.array([
                -W/2 + (i + 0.5) * w,  # x‐coordinate
                0,                     # y‐coordinate (middle of screen)
                0                      # z
            ])
            for i in range(num_sections)
        ]
        return [Dot(pt) for pt in centers]
    
    
    def _create_paper_factor_matrix(self) -> np.ndarray:
        # Create the COO list
        row_indices = [0, 1, 2, 2, 2, 3, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8]
        col_indices = [0, 1, 0, 1, 2, 3, 4, 3, 4, 5, 0, 2, 5, 6, 1, 2, 4, 5, 6, 7, 3, 5, 6, 7, 8]
        #Define random values in the size of row_indices or col_indices
        values = np.random.randn(len(row_indices))

        #Create the dense ndarray matrix which 9x9
        dense_matrix = np.zeros((9, 9))
        for i in range(len(row_indices)):
            dense_matrix[row_indices[i], col_indices[i]] = values[i]

        return dense_matrix
    

    def _create_paper_sparse_matrix(self) -> np.ndarray:
        # Create the COO list
        row_indices = [0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8]
        col_indices = [0, 0, 1, 1, 2, 0, 3, 0, 2, 4, 4, 5, 5, 6, 1, 6, 7, 2, 4, 8]
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
    

    def _linear_sys_definition(self) -> Tex:
        q = MathTex(r"A", r"\vec{x}", "=", r"\vec{b}", color=self.text_color, font_size=self.label_font_size)
        return q


    def _create_eq_group(self, A_mat: np.ndarray) -> VGroup:
        #Create the dense matrix
        A_pattern = create_manim_Matrix(A_mat.shape[0], A_mat.shape[1], A_mat)
        A_word = MathTex(r"A", color=self.text_color, font_size=self.label_font_size)

        # Multiplication of A and x
        mult_sign = MathTex(r"\times", color=self.text_color, font_size=self.label_font_size)
        #Create dense vector x in Ax=b
        x_pattern = create_dense_column_vector(A_mat.shape[0], font_size=self.value_font_size)
        x_word = MathTex(r"x", color=self.text_color, font_size=self.label_font_size)

        #Create the equal sign
        equal_sign = MathTex(r"=", color=self.text_color, font_size=self.label_font_size)

        #Create dense vector b in Ax=b
        b_pattern = create_dense_column_vector(A_mat.shape[0], font_size=self.value_font_size)
        b_word = MathTex(r"b", color=self.text_color, font_size=self.label_font_size)

        dense_eq_math = VGroup(A_pattern, mult_sign, x_pattern, equal_sign, b_pattern).arrange(RIGHT)
        A_word.next_to(dense_eq_math[0], UP)
        x_word.next_to(dense_eq_math[2], UP)
        b_word.next_to(dense_eq_math[4], UP)
        dense_eq = VGroup(dense_eq_math, A_word, x_word, b_word)
        return dense_eq
    
    def _create_llt_group(self, L: np.ndarray, x_pattern: Tex, b_pattern: Tex) -> VGroup:
        # Create the lower and upper triangular matrices
        L_mat = create_lower_triangular_matrix(L)
        L_pattern = create_manim_Matrix(L_mat.shape[0], L_mat.shape[1], L_mat)
        
        Lt_mat = create_upper_triangular_matrix(L.T)
        Lt_pattern = create_manim_Matrix(Lt_mat.shape[0], Lt_mat.shape[1], Lt_mat)
        
        # Write LL^t
        mult_sign = MathTex(r"\times", color=self.text_color, font_size=self.label_font_size)
        equal_sign = MathTex(r"=", color=self.text_color, font_size=self.label_font_size)

        llt_math_eq = VGroup(L_pattern, mult_sign.copy(),
                            Lt_pattern, mult_sign.copy(), x_pattern.copy(), equal_sign.copy(),
                              b_pattern.copy()).arrange(RIGHT)
        
        # Add labels
        L_word = MathTex(r"L", color=self.text_color, font_size=self.label_font_size)
        Lt_word = MathTex(r"L^T", color=self.text_color, font_size=self.label_font_size)
        b_llt_word = MathTex(r"b", color=self.text_color, font_size=self.label_font_size)
        x_llt_word = MathTex(r"x", color=self.text_color, font_size=self.label_font_size)

        L_word.next_to(llt_math_eq[0], UP)
        Lt_word.next_to(llt_math_eq[2], UP)
        x_llt_word.next_to(llt_math_eq[4], UP)
        b_llt_word.next_to(llt_math_eq[6], UP)

        llt_eq = VGroup(llt_math_eq, L_word, Lt_word, x_llt_word, b_llt_word)
        return llt_eq

    def _create_forward_group(self, L: np.ndarray, x_pattern: Tex, b_pattern: Tex) -> VGroup:
        # Create the lower triangular matrix
        L_pattern = create_matrix_tex_pattern(L.shape[0], L.shape[1], L, font_size=self.value_font_size)

        # Create the equal sign
        # Write LL^t
        mult_sign = MathTex(r"\times", color=self.text_color, font_size=self.label_font_size)
        equal_sign = MathTex(r"=", color=self.text_color, font_size=self.label_font_size)

        forward_math_eq = VGroup(L_pattern, mult_sign.copy(), x_pattern.copy(), equal_sign.copy(),
                              b_pattern.copy()).arrange(RIGHT)
        
        # Add labels
        L_word = MathTex(r"L", color=self.text_color, font_size=self.label_font_size)
        x_prime_forward_word = MathTex(r"x^{'}", color=self.text_color, font_size=self.label_font_size)
        b_foward_word = MathTex(r"b", color=self.text_color, font_size=self.label_font_size)

        L_word.next_to(forward_math_eq[0], UP)
        x_prime_forward_word.next_to(forward_math_eq[2], UP)
        b_foward_word.next_to(forward_math_eq[4], UP)

        forward_eq = VGroup(forward_math_eq, L_word, x_prime_forward_word, b_foward_word)
        #Add brace label
        forward_brace_label = BraceLabel(forward_eq, text=r"\text{Forward Substitution}", buff=0.1, font_size=self.label_font_size).set_color(self.text_color)
        forward_group = VGroup(forward_eq, forward_brace_label)

        return forward_group

    def _create_backward_group(self, Lt: np.ndarray, x_pattern: Tex, b_pattern: Tex) -> VGroup:
        # Create the lower triangular matrix
        Lt_pattern = create_matrix_tex_pattern(Lt.shape[0], Lt.shape[1], Lt, font_size=self.value_font_size)

        # Create the equal sign
        # Write LL^t
        mult_sign = MathTex(r"\times", color=self.text_color, font_size=self.label_font_size)
        equal_sign = MathTex(r"=", color=self.text_color, font_size=self.label_font_size)

        backward_math_eq = VGroup(Lt_pattern, mult_sign.copy(), x_pattern.copy(), equal_sign.copy(),
                              b_pattern.copy()).arrange(RIGHT)
        
        # Add labels
        Lt_word = MathTex(r"L^T", color=self.text_color, font_size=self.label_font_size)
        x_backward_word = MathTex(r"x", color=self.text_color, font_size=self.label_font_size)
        b_backward_word = MathTex(r"x^{'}", color=self.text_color, font_size=self.label_font_size)

        Lt_word.next_to(backward_math_eq[0], UP)
        x_backward_word.next_to(backward_math_eq[2], UP)
        b_backward_word.next_to(backward_math_eq[4], UP)

        backward_eq = VGroup(backward_math_eq, Lt_word, x_backward_word, b_backward_word)
        #Add brace label
        backward_brace_label = BraceLabel(backward_eq, text=r"\text{Backward Substitution}", buff=0.1, font_size=self.label_font_size).set_color(self.text_color)
        backward_group = VGroup(backward_eq, backward_brace_label)

        return backward_group
    
    def _create_solver_pipeline(self):
        # Linear system definition
        self.linear_sys_definition = self._linear_sys_definition()
        self.linear_sys_definition.center()

        A_mat = create_dense_matrix(self.size)
        self.dense_eq = self._create_eq_group(A_mat)
        self.dense_llt_eq = self._create_llt_group(A_mat, self.dense_eq[0][2], self.dense_eq[0][4])
        self.dense_forward_eq = self._create_forward_group(A_mat, self.dense_eq[0][2], self.dense_eq[0][4])
        self.dense_backward_eq = self._create_backward_group(A_mat, self.dense_eq[0][2], self.dense_eq[0][4])
        self.dense_backward_eq.next_to(self.dense_forward_eq, DOWN, buff=0.5)
        self.dense_forward_backward_eq = VGroup(self.dense_forward_eq, self.dense_backward_eq)
        self.dense_forward_backward_brace = Brace(self.dense_forward_backward_eq, direction=LEFT, color=self.text_color)
        self.dense_forward_backward_block = VGroup(self.dense_forward_backward_eq, self.dense_forward_backward_brace)

        A_sp_mat = self._create_paper_sparse_matrix()
        L_sp_sparsity = self._create_paper_factor_matrix()
        self.sparse_eq = self._create_eq_group(A_sp_mat)
        self.sparse_llt_eq = self._create_llt_group(L_sp_sparsity, self.sparse_eq[0][2], self.sparse_eq[0][4])
        self.sparse_forward_eq = self._create_forward_group(L_sp_sparsity, self.sparse_eq[0][2], self.sparse_eq[0][4])
        self.sparse_backward_eq = self._create_backward_group(L_sp_sparsity.T, self.sparse_eq[0][2], self.sparse_eq[0][4])
        self.sparse_backward_eq.next_to(self.sparse_forward_eq, DOWN, buff=0.5)
        self.sparse_forward_backward_eq = VGroup(self.sparse_forward_eq, self.sparse_backward_eq)
        self.sparse_forward_backward_brace = Brace(self.sparse_forward_backward_eq, direction=LEFT, color=self.text_color)
        self.sparse_forward_backward_block = VGroup(self.sparse_forward_backward_eq, self.sparse_forward_backward_brace)

    def _create_coo_scene_object(self) -> VGroup:
        row_indices = [0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8]
        col_indices = [0, 0, 1, 1, 2, 0, 3, 0, 2, 4, 4, 5, 5, 6, 1, 6, 7, 2, 4, 8]
        # Create a list of random numbers between 1 and 10 with 1 decimal point precision
        values = [round(np.random.uniform(1.0, 10.0), 1) for _ in range(len(row_indices))]

        #Create a three row vector of row_indices, col_indices, and values
        row_vec = np.array(row_indices).reshape(1, -1)
        col_vec = np.array(col_indices).reshape(1, -1)
        values_vec = np.array(values).reshape(1, -1)

        row_vec_pattern = create_matrix_tex_with_values(1, len(row_indices), row_vec, font_size=self.value_font_size)
        col_vec_pattern = create_matrix_tex_with_values(1, len(col_indices), col_vec, font_size=self.value_font_size)
        values_vec_pattern = create_matrix_tex_with_values(1, len(values), values_vec, font_size=self.value_font_size)

        row_vec_label = MathTex(r"\text{row indices}", color=self.text_color, font_size=self.label_font_size)
        col_vec_label = MathTex(r"\text{col indices}", color=self.text_color, font_size=self.label_font_size)
        values_vec_label = MathTex(r"\text{values}", color=self.text_color, font_size=self.label_font_size)

        coo_values = VGroup(row_vec_pattern, col_vec_pattern, values_vec_pattern).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        
        coo_labels = VGroup(row_vec_label, col_vec_label, values_vec_label).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        coo_labels.next_to(coo_values, LEFT)
        
        coo_scene_object = VGroup(coo_values, coo_labels)

        boxed = LabeledBox(coo_scene_object, "COO", stroke_color=self.text_color, label_font_size=self.label_font_size)
        
        return boxed


    def _create_csr_scene_object(self) -> VGroup:
        row_indices = [0, 1, 3, 5, 7, 10, 12, 14, 17, 20]
        col_indices = [0, 0, 1, 1, 2, 0, 3, 0, 2, 4, 4, 5, 5, 6, 1, 6, 7, 2, 4, 8]
        # Create a list of random numbers between 1 and 10 with 1 decimal point precision
        values = [round(np.random.uniform(1.0, 10.0), 1) for _ in range(len(col_indices))]

        #Create a three row vector of row_indices, col_indices, and values
        row_vec = np.array(row_indices).reshape(1, -1)
        col_vec = np.array(col_indices).reshape(1, -1)
        values_vec = np.array(values).reshape(1, -1)

        row_vec_pattern = create_matrix_tex_with_values(1, len(row_indices), row_vec, font_size=self.value_font_size)
        col_vec_pattern = create_matrix_tex_with_values(1, len(col_indices), col_vec, font_size=self.value_font_size)
        values_vec_pattern = create_matrix_tex_with_values(1, len(values), values_vec, font_size=self.value_font_size)

        row_vec_label = MathTex(r"\text{row pointer}", color=self.text_color, font_size=self.label_font_size)
        col_vec_label = MathTex(r"\text{col indices}", color=self.text_color, font_size=self.label_font_size)
        values_vec_label = MathTex(r"\text{values}", color=self.text_color, font_size=self.label_font_size)

        coo_values = VGroup(row_vec_pattern, col_vec_pattern, values_vec_pattern).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        
        coo_labels = VGroup(row_vec_label, col_vec_label, values_vec_label).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        coo_labels.next_to(coo_values, LEFT)
        
        coo_scene_object = VGroup(coo_values, coo_labels)

        boxed = LabeledBox(coo_scene_object, "CSR", stroke_color=self.text_color, label_font_size=self.label_font_size)
        
        return boxed

    def _extract_frame_index(self, filename: str) -> int:
        #the names are number.png 
        m = re.search(r"(\d+)", filename)
        if not m:
            raise ValueError(f"No digits in {filename!r}")
        return int(m.group(1))
    
    def _prepare_simulations_frames(self):
        # load & sort object renders
        imgs = [f for f in os.listdir("scripts/fix_sparsity_example/images/") if f.endswith(".png")]
        imgs.sort(key=self._extract_frame_index)
        self.frame_list = [
            ImageMobject(f"scripts/fix_sparsity_example/images/{f}") for f in imgs
        ]

        # load & sort matrix renders
        mats = [f for f in os.listdir("scripts/fix_sparsity_example/results/") if f.endswith(".png")]
        mats.sort(key=self._extract_frame_index)
        self.matrix_list = [
            ImageMobject(f"scripts/fix_sparsity_example/results/{f}") for f in mats
        ]
        min_size = min(len(self.frame_list), len(self.matrix_list))
        min_size = 50
        self.sim_frames = [self._show_simulation_frames(i) for i in range(min_size)]
    
    def _final_example(self)->Group:
        nefertiti_obj = ImageMobject(os.path.join("Figures", 'Background', 'nefertiti.png')).scale(1.2)
        sparse_obj = ImageMobject(os.path.join("Figures", 'Background', 'sparse_laplace.png'))
        sparse_obj.scale_to_fit_height(nefertiti_obj.get_height())
        nefertiti_obj.next_to(sparse_obj, RIGHT, buff=1)
        sparse_label = BraceLabel(sparse_obj, label_constructor=Tex, text="1M $\times$ 1M and 7M non-zeros (99.99\% sparsity)", buff=0.1, font_size=self.label_font_size, font_type=FONT_TYPE).set_color(self.text_color)
        example_group = Group(nefertiti_obj, sparse_obj, sparse_label).scale_to_fit_width(12)
        return example_group
 

    def _show_simulation_frames(self, iteration: int)->Group:
        #Create a VGroup of the frame and the matrix
        frame = self.frame_list[iteration]
        matrix = self.matrix_list[iteration]
        frame.scale(1.2)
        matrix.scale_to_fit_width(frame.get_width())
        matrix.next_to(frame, LEFT, buff=1)
        #Adding a surronding box around the matrix
        matrix_box = SurroundingRectangle(matrix, buff=0.0, color=BLACK, stroke_width=1)
        hessian_label = BraceLabel(matrix_box, text="Laplace Operator", label_constructor=Tex, buff=0.1, font_size=32).set_color(BLACK)
        mat_on_board_label = BraceLabel(frame, text="LibIgl Smoothing Example", label_constructor=Tex, buff=0.1, font_size=32).set_color(BLACK)
        return Group(frame, matrix, matrix_box, hessian_label, mat_on_board_label).scale_to_fit_height(6)
    
    def _final_example(self)->Group:
        nefertiti_obj = ImageMobject(os.path.join("Figures", 'Background', 'nefertiti.png')).scale(1.2)
        sparse_obj = ImageMobject(os.path.join("Figures", 'Background', 'sparse_laplace.png'))
        sparse_obj.scale_to_fit_height(nefertiti_obj.get_height())
        nefertiti_obj.next_to(sparse_obj, RIGHT, buff=1)
        sparse_label = BraceLabel(sparse_obj, text=r"\text{1M * 1M and 7M non-zeros (99.99\% sparsity)}", buff=0.1, font_size=self.label_font_size).set_color(self.text_color)
        example_group = Group(nefertiti_obj, sparse_obj, sparse_label)
        return example_group
 
    def play_background(self):
        self._create_solver_pipeline()

        # Entry point for Section 1 animation, with or without voiceover
        if isinstance(self.scene, VoiceoverScene):
            self._prepare_simulations_frames()
            script1 = "Many applications in geometric processing and physics simulation require solving a sparse symmetric semi-positive definite system of linear equations."
            with self.scene.voiceover(text=script1) as tracker:
                frame_and_matrix = self.sim_frames[0]
                frame_and_matrix.center()
                self.scene.play(FadeIn(self.sim_frames[0]), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

            script2 = "For example, here we can smooth out a mesh by successive application of the Laplacian matrix.\
                Where a linear system of equations is solved at each step."
            with self.scene.voiceover(text=script2) as tracker:
                total_time = tracker.duration
                time_per_iteration = total_time / (len(self.frame_list) - 1)
                rt = max(0.2, time_per_iteration)
                for i in range(1, len(self.sim_frames)):
                    new_frame_and_matrix = self.sim_frames[i]
                    new_frame_and_matrix.center()
                    self.scene.remove(frame_and_matrix)
                    self.scene.add(new_frame_and_matrix)
                    self.scene.wait(rt)
                    frame_and_matrix = new_frame_and_matrix

            script3 = "Also note that while the sparsity pattern is constant, the values are changing."
            with self.scene.voiceover(text=script3) as tracker:
                pass

            script4 = "To have fast linear solvers for these problems, state-of-the-art tools provide symbolic-numeric framework."
            with self.scene.voiceover(text=script4) as tracker:
                pass

            sparse_matrix = create_sparse_matrix(9, 0, 0.1)
            solver = SymbolicNumericFramework(A_sp=sparse_matrix, matrix_size=9,
                                                            generate_random_pattern=True, generate_random_values=True,
                                                            matrix_name="H", rhs_name="-g", unknown_name="d")
            solver.center()
            solver_internal = solver[0]
            symbolic_input = solver_internal[0]
            arrow = solver_internal[1]
            symbolic_box = solver_internal[2][0]
            symbolic_section = VGroup(symbolic_input, arrow, symbolic_box)
            script5 = "In this framework, first, the symbolic analysis phase is performed to analyze the sparsity pattern of the matrix."
            with self.scene.voiceover(text=script5) as tracker:
                self.scene.wait(self.wait_time)
                self.scene.play(FadeOut(frame_and_matrix), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)
                self.scene.play(FadeIn(symbolic_section), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

            sym_to_numeric_arrow = solver_internal[2][1]
            numeric_box = solver_internal[2][2]
            numeric_input = solver_internal[3]
            numeric_arrow = solver_internal[4]
            arrow_to_solve = solver_internal[5]
            solve_value = solver_internal[6]
            label = solver_internal[7]
            numeric_section = VGroup(sym_to_numeric_arrow, numeric_box, numeric_input, numeric_arrow, arrow_to_solve, solve_value, label)
            script6 = "Then, using the symbolic analysis results, the numerical computation phase is efficiently performed to solve the linear system of equations."
            with self.scene.voiceover(text=script6) as tracker:
                self.scene.play(FadeIn(numeric_section), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

            script6_1 = "In general symbolic analysis can be more expensive than the numerical computation phase.\
                Here for laplace operator, the symbolic analysis takes 1.9s while the numerical computation takes 1s."
            with self.scene.voiceover(text=script6_1) as tracker:
                self.scene.play(FadeOut(solver), run_time=self.transform_runtime)
                final_example = self._final_example()
                final_example.center()
                self.scene.play(FadeIn(final_example), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

            script7 = "However if the sparsity pattern remains constant, the symbolic analysis can be reused,\
                amortizing the expensive symbolic analysis overhead across multiple numerical calls."
            with self.scene.voiceover(text=script7) as tracker:
                self.scene.play(FadeOut(final_example), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)
                total_time = tracker.duration
                time_per_iteration = total_time / 10
                smile_emoji = ImageMobject("Figures/Problem/smile.png").scale(0.3)
                smile_emoji.next_to(solver[0][0], UP, buff=1)
                self.scene.play(FadeIn(smile_emoji), run_time=self.transform_runtime)
                for i in range(10):
                    numeric_color = RED_A if i % 2 == 0 else GREEN_A
                    new_framework = SymbolicNumericFramework(A_sp=sparse_matrix, iteration=i, numeric_box_color=numeric_color,
                                                            matrix_size=9, generate_random_pattern=False, generate_random_values=True,
                                                            matrix_name="H", rhs_name="-g", unknown_name="d")
                    new_framework.move_to(solver.get_center())
                    self.scene.play(Transform(solver, new_framework), run_time=time_per_iteration)




    def play_cow_scene(self, scene: Scene | VoiceoverScene):
        self.scene = scene
        self._prepare_simulations_frames()

        #Scale each frame to fit the whole screen
        for frame in self.sim_frames:
            frame.scale_to_fit_width(self.scene.camera.frame_width * 0.95)

        frame_and_matrix = self.sim_frames[0]
        frame_and_matrix.center()
        self.scene.add(frame_and_matrix)

        script = "For example, here we can smooth out a mesh by successive application of the Laplacian matrix, where a linear system of equations is solved at each step."
        with self.scene.voiceover(text=script) as tracker:
            total_time = tracker.duration
            time_per_iteration = total_time / (len(self.frame_list) - 1)
            rt = max(0.2, time_per_iteration)
            for i in range(1, len(self.sim_frames)):
                new_frame_and_matrix = self.sim_frames[i]
                new_frame_and_matrix.center()
                self.scene.remove(frame_and_matrix)
                self.scene.add(new_frame_and_matrix)
                self.scene.wait(rt)
                frame_and_matrix = new_frame_and_matrix


    def play_framework_introduction_scene(self, scene: Scene | VoiceoverScene):
        self.scene = scene
        sparse_matrix = create_sparse_matrix(9, 0, 0.1)
        solver = SymbolicNumericFramework(A_sp=sparse_matrix, matrix_size=9,
                                                        generate_random_pattern=True, generate_random_values=True,
                                                        matrix_name="H", rhs_name="-g", unknown_name="d")
        solver.scale_to_fit_width(self.scene.camera.frame_width * 0.95)
        solver.center()
        solver_internal = solver[0]
        symbolic_input = solver_internal[0]
        arrow = solver_internal[1]
        symbolic_box = solver_internal[2][0]
        symbolic_section = VGroup(symbolic_input, arrow, symbolic_box)

        # script0 = "To have fast Cholesky solvers for these problems, state-of-the-art tools provide a symbolic-numeric framework"
        # with self.scene.voiceover(text=script0) as tracker:
        #     pass
        
        script1 = "In this framework, the first step is to perform the symbolic analysis phase, which involves analyzing the sparsity pattern of the matrix."
        with self.scene.voiceover(text=script1) as tracker:
            self.scene.play(FadeIn(symbolic_section), run_time=self.transform_runtime)
            self.scene.wait(self.wait_time)

        sym_to_numeric_arrow = solver_internal[2][1]
        numeric_box = solver_internal[2][2]
        numeric_input = solver_internal[3]
        numeric_arrow = solver_internal[4]
        # arrow_to_solve = solver_internal[5]
        # solve_value = solver_internal[6]
        # label = solver_internal[7]
        numeric_section = VGroup(sym_to_numeric_arrow, numeric_box, numeric_input, numeric_arrow)
        script2 = "Then, using the symbolic analysis results, the numerical computation phase is efficiently performed to solve the linear system of equations."
        with self.scene.voiceover(text=script2) as tracker:
            self.scene.play(FadeIn(numeric_section), run_time=self.transform_runtime)
            self.scene.wait(self.wait_time)

    
    def play_example_of_expensive_symbolic_analysis(self, scene: Scene | VoiceoverScene):
        final_example = self._final_example()
        final_example.scale_to_fit_width(self.scene.camera.frame_width * 0.95)
        final_example.center()
        self.scene.add(final_example)


    def play_computational_flow_scene(self, scene: Scene | VoiceoverScene):
        sparse_matrix = create_sparse_matrix(9, 0, 0.1)
        solver = SymbolicNumericFramework(A_sp=sparse_matrix, matrix_size=9,
                                                generate_random_pattern=False, generate_random_values=False,
                                                matrix_name="H", rhs_name="-g", unknown_name="d")
        solver.scale_to_fit_width(self.scene.camera.frame_width * 0.95)
        solver.center()
        self.scene.add(solver)
        
        total_time = 10
        time_per_iteration = total_time / 10
        script = "In this example, after a single symbolic analysis run, its overhead is amortized over multiple numerical calls."
        with self.scene.voiceover(text=script) as tracker:
            for i in range(10):
                numeric_color = RED_A if i % 2 == 0 else GREEN_A
                new_framework = SymbolicNumericFramework(A_sp=sparse_matrix, iteration=i, numeric_box_color=numeric_color,
                                                        matrix_size=9, generate_random_pattern=False, generate_random_values=True,
                                                        matrix_name="H", rhs_name="-g", unknown_name="d")
                new_framework.move_to(solver.get_center())
                self.scene.play(Transform(solver, new_framework), run_time=time_per_iteration)

            smile_emoji = ImageMobject("Figures/Problem/smile.png").scale(0.3)
            smile_emoji.next_to(solver[0][0], UP, buff=1)
            self.scene.play(FadeIn(smile_emoji), run_time=self.transform_runtime)




    