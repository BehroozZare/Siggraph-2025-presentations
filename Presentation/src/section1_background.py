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

# 2) extend the pre-amble
template = TexTemplate()
template.add_to_preamble(r"\usepackage{xcolor}")

config.tex_template = template


# Section1: Handles the animation and logic for the problem definition section of the FastTrack video.
class Background():
    def __init__(self, scene: Scene | VoiceoverScene):
        # Initialize the section with timing and animation parameters
        self.scene = scene
        self.size = 9
        self.text_color = BLACK
        self.scale_factor = 0.6
        self.label_font_size = 32
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
        A_pattern = create_matrix_tex_pattern(A_mat.shape[0], A_mat.shape[1], A_mat, font_size=self.value_font_size)
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
        L_pattern = create_matrix_tex_pattern(L_mat.shape[0], L_mat.shape[1], L_mat, font_size=self.value_font_size)
        
        Lt_mat = create_upper_triangular_matrix(L.T)
        Lt_pattern = create_matrix_tex_pattern(Lt_mat.shape[0], Lt_mat.shape[1], Lt_mat, font_size=self.value_font_size)
        
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
        centers = self._get_centers_of_section(3)

        # Entry point for Section 1 animation, with or without voiceover
        if isinstance(self.scene, VoiceoverScene):
            # scripts = "Solving a symmetric semi-positive definite system of linear equations is a core numerical task in many applications. \
            # For accurate computations of the solution 'X', a direct methods such Cholesky factorization is used. \
            # The Cholesky factorization decompose the matrix into the multiplecation of a lower triangular matrix and its transpose. \
            # Followed by a forward and backward substitution to solve for the solution 'X'. \
            # Often time, in applications such as those using Finite Element methods, The matrix A is sparse.\
            # In these applications, approximately more than 95\% of the entries are zero.\
            # To exploit this sparsity, the matrix is stored in sparse formats, such as Coordinate list, or compressed row-format.\
            # To have fast codes with these format, state-of-the-art sparse solvers employ two steps.\
            # First, the sparsity pattern of matrix A is analyzed in the phase called symbolic analysis. Where depending on the sparsity \
            # a set of acceleration techniques are selected and used for faster execution of numerical computations. \
            # Second, the numerical computations are performed in the phase called numeric analysis. \
            # While the symbolic analysis is often time more expensive than the executor, this two-step approach is justifiable, as in \
            # these applications, often time the symbolic analysis is only performed once, and the numeric analysis is performed multiple times."
            script1 = "Solving systems of linear equations is a common task in many computational applications."
            arrow_system_to_factor = None
            arrow_factor_to_forward_backward = None
            arrow_sparse_to_coo = None
            with self.scene.voiceover(text=script1) as tracker:    
                self.scene.play(Write(self.linear_sys_definition), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)
                self.dense_eq.move_to(self.linear_sys_definition.get_center())
                self.scene.play(ReplacementTransform(self.linear_sys_definition, self.dense_eq), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

            script2 = "Here we are focusing on Cholesky factorization, which decomposes the matrix into lower and upper triangular matrices."
            with self.scene.voiceover(text=script2) as tracker:
                # Scale and move up
                self.scene.play(self.dense_eq.animate.scale(self.scale_factor).move_to(centers[0]), run_time=self.transform_runtime)

                self.dense_llt_eq.scale(self.scale_factor)
                self.dense_llt_eq.move_to(centers[1])

                #Create an arrow from dense to sparse llt eq
                arrow_system_to_factor = Arrow(self.dense_eq.get_right(), self.dense_llt_eq.get_left(), stroke_width=self.arrow_stroke_width, color=self.text_color)
                self.scene.play(Create(arrow_system_to_factor), run_time=self.arrow_runtime)
                self.scene.play(FadeIn(self.dense_llt_eq), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

                #Create an arrow to forward_backward_block
                self.dense_forward_backward_block.scale(self.scale_factor)
                self.dense_forward_backward_block.move_to(centers[2])


            script3 = "The decomposition is followed by forward and backward substitution to compute the solution"
            with self.scene.voiceover(text=script3) as tracker:
                arrow_factor_to_forward_backward = Arrow(self.dense_llt_eq.get_right(), self.dense_forward_backward_block.get_left(), stroke_width=self.arrow_stroke_width, color=self.text_color)
                self.scene.play(Create(arrow_factor_to_forward_backward), run_time=self.arrow_runtime)
                self.scene.play(FadeIn(self.dense_forward_backward_block), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

            script4 = "For Finite Element Method, the linear system is often symmetric and sparse."
            with self.scene.voiceover(text=script4) as tracker:
                #Sparse pipeline
                self.sparse_eq.move_to(self.dense_eq.get_center())
                self.sparse_eq.scale(self.scale_factor)

                self.sparse_llt_eq.move_to(self.dense_llt_eq.get_center())
                self.sparse_llt_eq.scale(self.scale_factor)

                self.sparse_forward_backward_block.move_to(self.dense_forward_backward_block.get_center())
                self.sparse_forward_backward_block.scale(self.scale_factor)

                self.scene.wait(self.wait_time)
                self.scene.play(ReplacementTransform(self.dense_eq, self.sparse_eq),
                                ReplacementTransform(self.dense_llt_eq, self.sparse_llt_eq),
                                ReplacementTransform(self.dense_forward_backward_block, self.sparse_forward_backward_block), run_time=self.transform_runtime)
                sparse_matrix_label = BraceLabel(self.sparse_eq, text=r"\text{More than 95\% Sparse}", buff=0.1, font_size=self.label_font_size).set_color(self.text_color)
                sparse_matrix_label.next_to(self.sparse_eq[0][0], DOWN)
                self.scene.wait(self.wait_time)
                self.scene.play(FadeIn(sparse_matrix_label), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

            script5 = "To take advantage of the sparsity, sparse matrix formats such as Compressed Row format are proposed."
            with self.scene.voiceover(text=script5) as tracker:
                # Fadeout all the objects beside the linear system
                sparse_linear_system = self.sparse_eq[0][0]
                sparse_linear_system_label = self.sparse_eq[1]
                A_matrix = VGroup(sparse_linear_system, sparse_linear_system_label)
                self.scene.play(FadeOut(self.sparse_eq[0][1]),
                                FadeOut(self.sparse_eq[0][2]),
                                FadeOut(self.sparse_eq[0][3]),
                                FadeOut(self.sparse_eq[0][4]),
                                FadeOut(self.sparse_eq[2]),
                                FadeOut(self.sparse_eq[3]),
                                FadeOut(arrow_system_to_factor),
                                FadeOut(self.sparse_llt_eq),
                                FadeOut(arrow_factor_to_forward_backward),
                                FadeOut(self.sparse_forward_backward_block),
                                FadeOut(sparse_matrix_label),
                                run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

                
                self.scene.play(A_matrix.animate.scale(1 / self.scale_factor).to_edge(LEFT, buff=2), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)
            
                csr_format_object = self._create_csr_scene_object()
                csr_format_object.to_edge(RIGHT, buff=2)
                
                #Brace label
                sparse_brace_label = BraceLabel(csr_format_object, text=r"\text{Sparse Format}", buff=0.1, font_size=self.label_font_size).set_color(self.text_color)
                dense_brace_label = BraceLabel(A_matrix, text=r"\text{Dense Format}", buff=0.1, font_size=self.label_font_size).set_color(self.text_color)

                self.scene.play(FadeIn(csr_format_object), run_time=self.transform_runtime)
                self.scene.play(FadeIn(sparse_brace_label), FadeIn(dense_brace_label), run_time=self.transform_runtime)

            script6 = "These formats save memory. However, applying parallelism and vectorization become non-trivial.\
            Here, for example, retrieving an element from a dense layout is straightforward, but in a sparse one, you must scan multiple non-zero entries."
            with self.scene.voiceover(text=script6) as tracker:
                #Fade out everythings beside the csr_Format_object
                dense_ret_code = Code(
                    code_file="Materials/dense.cpp",
                    tab_width=4,
                    language="C++",
                    background="rectangle",
                    add_line_numbers=False,
                    formatter_style="monokai",
                ).scale(0.8)
                sparse_ret_code = Code(
                    code_file="Materials/sparse.cpp",
                    tab_width=4,
                    language="C++",
                    background="rectangle",
                    add_line_numbers=False,
                    formatter_style="monokai",
                ).scale(0.8)
                dense_ret_code.next_to(dense_brace_label, UP)
                sparse_ret_code.next_to(sparse_brace_label, UP)

                code_sparse_brace_label = BraceLabel(sparse_ret_code, text=r"\text{Sparse Format}", buff=0.1, font_size=self.label_font_size).set_color(self.text_color)
                code_dense_brace_label = BraceLabel(dense_ret_code, text=r"\text{Dense Format}", buff=0.1, font_size=self.label_font_size).set_color(self.text_color)


                self.scene.play(FadeOut(csr_format_object),
                                FadeOut(A_matrix),
                                run_time=self.transform_runtime)
                
                self.scene.play(FadeIn(dense_ret_code), FadeIn(sparse_ret_code),
                                ReplacementTransform(sparse_brace_label, code_sparse_brace_label),
                                ReplacementTransform(dense_brace_label, code_dense_brace_label), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

            if self.scene.mobjects:
                self.scene.play(FadeOut(*self.scene.mobjects), run_time=0.2)
                self.scene.wait(0.1)

            script7 = "State-of-the-art sparse solvers use a two-step approach to reduce this limitation.\
            In symbolic phase, the sparsity pattern is analyzed for applying acceleration techniques. \
            Then, using this analysis, the numerical phase is performed, where the factorization and forward/backward substitution are performed."
            with self.scene.voiceover(text=script7) as tracker:
                #Create two objects 
                A_sp = self._create_paper_sparse_matrix()
                symbolic_numeric_framework = SymbolicNumericFramework(A_sp)
                symbolic_numeric_framework.center()
                self.scene.play(Create(symbolic_numeric_framework), run_time=self.transform_runtime)


            script8 = "Here as an example, we apply laplace-beltrami operator on this mesh.\
                The resultant matrix has approximately 7 million non-zeros with more than 99% sparsity.\
                The symbolic analysis is performed in 1.9 seconds and the numeric computation is performed in 1 second.\
                Note that in here, the symbolic analysis runtime is more than numeric computation for a single Cholesky solve."
            with self.scene.voiceover(text=script8) as tracker:
                final_example = self._final_example()
                final_example.center()
                self.scene.play(FadeOut(symbolic_numeric_framework), run_time=self.transform_runtime)
                self.scene.play(FadeIn(final_example), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)
                
        
        else:
            self._play_scene()



#cp 12_matOnBoard_seg0/numThreads_20_SolverType_CHOLMOD_IM_0/hessian_*_0_last_IPC.mtx hessian_checkpoints/