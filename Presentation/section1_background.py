from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import cholesky

# 2) extend the pre-amble
template = TexTemplate()
template.add_to_preamble(r"\usepackage{xcolor}")

config.tex_template = template


# Section1: Handles the animation and logic for the problem definition section of the FastTrack video.
class Section1():
    def __init__(self, scene: Scene | VoiceoverScene):
        # Initialize the section with timing and animation parameters
        self.scene = scene
        self.text_color = BLACK
        self.scale_factor = 0.4
        self.transform_runtime = 0.5
        self.forward_backward_creation_runtime = 2
        self.arrow_runtime = 0.5
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


    def _create_lower_triangular_matrix(self, matrix: np.ndarray) -> np.ndarray:
        # Create a lower triangular dense matrix pattern for visualization
        lower_triangular_matrix = np.tril(matrix)
        return lower_triangular_matrix
    
    def _create_upper_triangular_matrix(self, matrix: np.ndarray) -> np.ndarray:
        # Create a upper triangular dense matrix pattern for visualization
        upper_triangular_matrix = np.triu(matrix)
        return upper_triangular_matrix
    

    def _create_sparse_matrix(self, size: int, iteration: int, sparsity: float) -> np.ndarray:
        # Create a symmetric sparse matrix pattern for visualization
        
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

        return matrix_data
    
    def _create_dense_matrix(self, size: int) -> np.ndarray:
        # Create a dense matrix pattern for visualization
        matrix_data = np.random.randn(size, size)
        return matrix_data


    def _create_matrix_tex_pattern(self, row_num: int, col_num: int, matrix: np.ndarray, font_size: int = 36) -> Tex:
        # Create the LaTeX matrix string for visualization
        col_align = "|" + "c" * col_num + "|"
        matrix_str = r"$\begin{array}{" + col_align + r"}"
        for i in range(row_num):
            row_parts = []
            for j in range(col_num):
                if abs(matrix[i, j]) > 1e-10:  # Non-zero element
                    row_parts.append(r"\ast")
                else:  # Zero element
                    row_parts.append("")  # Empty string for zero elements
            matrix_str += " & ".join(row_parts)
            if i < row_num - 1:
                matrix_str += r" \\ "
        matrix_str += r"\end{array}$"
        
        
        # Create the matrix as LaTeX
        matrix_pattern = Tex(matrix_str, color=self.text_color, font_size=font_size)  # Reduced font size for better fit
        matrix_pattern.scale(0.5)  # Scale down for better visibility
        return matrix_pattern
    
    def _create_dense_column_vector(self, matrix_size: int) -> Tex:
        # Create a dense column vector pattern for visualization
        column_vector = np.random.randn(matrix_size, 1)
        column_vector_pattern = self._create_matrix_tex_pattern(matrix_size, 1, column_vector)
        return column_vector_pattern
    
    def _create_dense_row_vector(self, matrix_size: int) -> Tex:
        # Create a dense row vector pattern for visualization
        row_vector = np.random.randn(1, matrix_size)
        row_vector_pattern = self._create_matrix_tex_pattern(1, matrix_size, row_vector)
        return row_vector_pattern
    
    def _linear_sys_definition(self) -> Tex:
        q = MathTex(r"A", r"\vec{x}", "=", r"\vec{b}", color=self.text_color, font_size=36)
        return q
    
    def _dense_matrix_shape(self) -> Tex:
        q = MathTex(r"A", r"\vec{x}", "=", r"\vec{b}", color=self.text_color, font_size=36)
        return q
    
    def _dense_lower_triangular_matrix(self) -> Tex:
        q = MathTex(r"L", color=self.text_color, font_size=36)
        return q
    
    def _dense_upper_triangular_matrix(self) -> Tex:
        q = MathTex(r"U", color=self.text_color, font_size=36)
        return q
    
    def _dense_forward_substitution(self) -> Tex:
        pass

    def _dense_backward_substitution(self) -> Tex:
        pass

    def _sparsifying_dense_lower_upper_triangular_matrix(self) -> Tex:
        pass

    def _create_eq_group(self, A_mat: np.ndarray) -> VGroup:
        #Create the dense matrix
        A_pattern = self._create_matrix_tex_pattern(A_mat.shape[0], A_mat.shape[1], A_mat)
        A_word = MathTex(r"A", color=self.text_color, font_size=36)

        # Multiplication of A and x
        mult_sign = MathTex(r"\times", color=self.text_color, font_size=36)
        #Create dense vector x in Ax=b
        x_pattern = self._create_dense_column_vector(A_mat.shape[0])
        x_word = MathTex(r"x", color=self.text_color, font_size=36)

        #Create the equal sign
        equal_sign = MathTex(r"=", color=self.text_color, font_size=36)

        #Create dense vector b in Ax=b
        b_pattern = self._create_dense_column_vector(A_mat.shape[0])
        b_word = MathTex(r"b", color=self.text_color, font_size=36)

        dense_eq_math = VGroup(A_pattern, mult_sign, x_pattern, equal_sign, b_pattern).arrange(RIGHT)
        A_word.next_to(dense_eq_math[0], UP)
        x_word.next_to(dense_eq_math[2], UP)
        b_word.next_to(dense_eq_math[4], UP)
        dense_eq = VGroup(dense_eq_math, A_word, x_word, b_word)
        return dense_eq
    
    def _create_llt_group(self, L: np.ndarray, x_pattern: Tex, b_pattern: Tex) -> VGroup:
        # Create the lower and upper triangular matrices
        L_mat = self._create_lower_triangular_matrix(L)
        L_pattern = self._create_matrix_tex_pattern(L_mat.shape[0], L_mat.shape[1], L_mat)
        
        Lt_mat = self._create_upper_triangular_matrix(L.T)
        Lt_pattern = self._create_matrix_tex_pattern(Lt_mat.shape[0], Lt_mat.shape[1], Lt_mat)
        
        # Write LL^t
        mult_sign = MathTex(r"\times", color=self.text_color, font_size=36)
        equal_sign = MathTex(r"=", color=self.text_color, font_size=36)

        llt_math_eq = VGroup(L_pattern, mult_sign.copy(),
                            Lt_pattern, mult_sign.copy(), x_pattern.copy(), equal_sign.copy(),
                              b_pattern.copy()).arrange(RIGHT)
        
        # Add labels
        L_word = MathTex(r"L", color=self.text_color, font_size=36)
        Lt_word = MathTex(r"L^T", color=self.text_color, font_size=36)
        b_llt_word = MathTex(r"b", color=self.text_color, font_size=36)
        x_llt_word = MathTex(r"x", color=self.text_color, font_size=36)

        L_word.next_to(llt_math_eq[0], UP)
        Lt_word.next_to(llt_math_eq[2], UP)
        x_llt_word.next_to(llt_math_eq[4], UP)
        b_llt_word.next_to(llt_math_eq[6], UP)

        llt_eq = VGroup(llt_math_eq, L_word, Lt_word, x_llt_word, b_llt_word)
        return llt_eq


    def _create_forward_group(self, L: np.ndarray, x_pattern: Tex, b_pattern: Tex) -> VGroup:
        # Create the lower triangular matrix
        L_pattern = self._create_matrix_tex_pattern(L.shape[0], L.shape[1], L)

        # Create the equal sign
        # Write LL^t
        mult_sign = MathTex(r"\times", color=self.text_color, font_size=36)
        equal_sign = MathTex(r"=", color=self.text_color, font_size=36)

        forward_math_eq = VGroup(L_pattern, mult_sign.copy(), x_pattern.copy(), equal_sign.copy(),
                              b_pattern.copy()).arrange(RIGHT)
        
        # Add labels
        L_word = MathTex(r"L", color=self.text_color, font_size=36)
        x_prime_forward_word = MathTex(r"x^{'}", color=self.text_color, font_size=36)
        b_foward_word = MathTex(r"b", color=self.text_color, font_size=36)

        L_word.next_to(forward_math_eq[0], UP)
        x_prime_forward_word.next_to(forward_math_eq[2], UP)
        b_foward_word.next_to(forward_math_eq[4], UP)

        forward_eq = VGroup(forward_math_eq, L_word, x_prime_forward_word, b_foward_word)
        #Add brace label
        forward_brace_label = BraceLabel(forward_eq, text=r"\text{Forward Substitution}", buff=0.1, font_size=36).set_color(self.text_color)
        forward_group = VGroup(forward_eq, forward_brace_label)

        return forward_group

    def _create_backward_group(self, Lt: np.ndarray, x_pattern: Tex, b_pattern: Tex) -> VGroup:
        # Create the lower triangular matrix
        Lt_pattern = self._create_matrix_tex_pattern(Lt.shape[0], Lt.shape[1], Lt)

        # Create the equal sign
        # Write LL^t
        mult_sign = MathTex(r"\times", color=self.text_color, font_size=36)
        equal_sign = MathTex(r"=", color=self.text_color, font_size=36)

        backward_math_eq = VGroup(Lt_pattern, mult_sign.copy(), x_pattern.copy(), equal_sign.copy(),
                              b_pattern.copy()).arrange(RIGHT)
        
        # Add labels
        Lt_word = MathTex(r"L^T", color=self.text_color, font_size=36)
        x_backward_word = MathTex(r"x", color=self.text_color, font_size=36)
        b_backward_word = MathTex(r"x^{'}", color=self.text_color, font_size=36)

        Lt_word.next_to(backward_math_eq[0], UP)
        x_backward_word.next_to(backward_math_eq[2], UP)
        b_backward_word.next_to(backward_math_eq[4], UP)

        backward_eq = VGroup(backward_math_eq, Lt_word, x_backward_word, b_backward_word)
        #Add brace label
        backward_brace_label = BraceLabel(backward_eq, text=r"\text{Backward Substitution}", buff=0.1, font_size=36).set_color(self.text_color)
        backward_group = VGroup(backward_eq, backward_brace_label)

        return backward_group
    
    def _create_solver_pipeline(self):
        # Linear system definition
        self.linear_sys_definition = self._linear_sys_definition()
        self.linear_sys_definition.center()

        A_mat = self._create_dense_matrix(16)
        self.dense_eq = self._create_eq_group(A_mat)
        self.dense_llt_eq = self._create_llt_group(A_mat, self.dense_eq[0][2], self.dense_eq[0][4])
        self.dense_forward_eq = self._create_forward_group(A_mat, self.dense_eq[0][2], self.dense_eq[0][4])
        self.dense_backward_eq = self._create_backward_group(A_mat, self.dense_eq[0][2], self.dense_eq[0][4])
        self.dense_backward_eq.next_to(self.dense_forward_eq, DOWN, buff=0.5)
        self.dense_forward_backward_eq = VGroup(self.dense_forward_eq, self.dense_backward_eq)
        self.dense_forward_backward_brace = Brace(self.dense_forward_backward_eq, direction=LEFT, color=self.text_color)
        self.dense_forward_backward_block = VGroup(self.dense_forward_backward_eq, self.dense_forward_backward_brace)

        A_sp_mat = self._create_sparse_matrix(16, 0, 0.1)
        L_sp_sparsity = self.cholesky_sparsity_pattern(A_sp_mat)
        self.sparse_eq = self._create_eq_group(A_sp_mat)
        self.sparse_llt_eq = self._create_llt_group(L_sp_sparsity, self.sparse_eq[0][2], self.sparse_eq[0][4])
        self.sparse_forward_eq = self._create_forward_group(L_sp_sparsity, self.sparse_eq[0][2], self.sparse_eq[0][4])
        self.sparse_backward_eq = self._create_backward_group(L_sp_sparsity.T, self.sparse_eq[0][2], self.sparse_eq[0][4])
        self.sparse_backward_eq.next_to(self.sparse_forward_eq, DOWN, buff=0.5)
        self.sparse_forward_backward_eq = VGroup(self.sparse_forward_eq, self.sparse_backward_eq)
        self.sparse_forward_backward_brace = Brace(self.sparse_forward_backward_eq, direction=LEFT, color=self.text_color)
        self.sparse_forward_backward_block = VGroup(self.sparse_forward_backward_eq, self.sparse_forward_backward_brace)




 
    def play_scene(self):
        self._create_solver_pipeline()
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
            script1 = "Solving a system of linear is a common computational task in many applications."
            with self.scene.voiceover(text=script1) as tracker:    
                self.scene.play(Write(self.linear_sys_definition), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

            script2 = "One of the way to have accurate solution, is by using Cholesky factorization. \
            Where the matrix is decomposed into a lower and upper trianglular matrix. This decomposition then \
                followed by a forward and backward substitution to solve for the solution 'X'."
            with self.scene.voiceover(text=script2) as tracker:
                self.dense_eq.move_to(self.linear_sys_definition.get_center())
                self.scene.play(ReplacementTransform(self.linear_sys_definition, self.dense_eq), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

                # Scale and move up
                self.scene.play(self.dense_eq.animate.scale(self.scale_factor).to_edge(LEFT), run_time=self.transform_runtime)

                self.dense_llt_eq.scale(self.scale_factor)
                self.dense_llt_eq.next_to(self.dense_eq, RIGHT, buff=1)

                #Create an arrow from dense to sparse llt eq
                arrow = Arrow(self.dense_eq.get_right(), self.dense_llt_eq.get_left(), stroke_width=2, color=self.text_color)
                self.scene.play(Create(arrow), run_time=self.arrow_runtime)
                self.scene.play(Create(self.dense_llt_eq), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

                #Create an arrow to forward_backward_block
                self.dense_forward_backward_block.scale(self.scale_factor)
                self.dense_forward_backward_block.next_to(self.dense_llt_eq, RIGHT, buff=1)
                arrow = Arrow(self.dense_llt_eq.get_right(), self.dense_forward_backward_block.get_left(), stroke_width=2, color=self.text_color)
                self.scene.play(Create(arrow), run_time=self.arrow_runtime)
                self.scene.play(Create(self.dense_forward_backward_block), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)


            script3 = "However in many applications, such as those involving Finite Element Methods, \
            the system is sparse. In these application, the zeros in linear system matrix, can comprise more than 95\% of the entries."
            with self.scene.voiceover(text=script3) as tracker:
                #Sparse pipeline
                self.sparse_eq.move_to(self.dense_eq.get_center())
                self.sparse_eq.scale(self.scale_factor)

                self.sparse_llt_eq.move_to(self.dense_llt_eq.get_center())
                self.sparse_llt_eq.scale(self.scale_factor)

                self.sparse_forward_backward_block.move_to(self.dense_forward_backward_block.get_center())
                self.sparse_forward_backward_block.scale(self.scale_factor)

                self.scene.play(ReplacementTransform(self.dense_eq, self.sparse_eq),
                                ReplacementTransform(self.dense_llt_eq, self.sparse_llt_eq),
                                ReplacementTransform(self.dense_forward_backward_block, self.sparse_forward_backward_block), run_time=self.transform_runtime)
            
        
        else:
            self._play_scene()