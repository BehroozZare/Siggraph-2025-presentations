from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import numpy as np

# 2) extend the pre-amble
template = TexTemplate()
template.add_to_preamble(r"\usepackage{xcolor}")

config.tex_template = template


# Section1: Handles the animation and logic for the problem definition section of the FastTrack video.
class Section2():
    def __init__(self, scene: Scene | VoiceoverScene):
        # Initialize the section with timing and animation parameters
        self.scene = scene
        self.transform_runtime = 0.5
        self.wait_time = 1

        self.dense_matrix = None
        self.dense_lower_triangular_matrix = None
        self.dense_upper_triangular_matrix = None
        self.sparse_matrix = None
        self.sparse_lower_triangular_matrix = None
        self.sparse_upper_triangular_matrix = None
        
        

    def _create_lower_triangular_matrix(self, matrix: np.ndarray) -> np.ndarray:
        # Create a lower triangular dense matrix pattern for visualization
        lower_triangular_matrix = np.tril(matrix)
        return lower_triangular_matrix
    
    def _create_upper_triangular_matrix(self, matrix: np.ndarray) -> np.ndarray:
        # Create a upper triangular dense matrix pattern for visualization
        upper_triangular_matrix = np.triu(matrix)
        return upper_triangular_matrix
    

    def _create_sparse_matrix(self, iteration: int, sparsity: float) -> np.ndarray:
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

        return matrix_data
    
    def _create_dense_matrix(self) -> np.ndarray:
        # Create a dense matrix pattern for visualization
        size = 16  # Keep size manageable for LaTeX
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
        matrix_pattern = Tex(matrix_str, font_size=font_size)  # Reduced font size for better fit
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
        q = MathTex(r"A", r"\vec{x}", "=", r"\vec{b}", font_size=36)
        return q
    
    def _dense_matrix_shape(self) -> Tex:
        q = MathTex(r"A", r"\vec{x}", "=", r"\vec{b}", font_size=36)
        return q
    
    def _dense_lower_triangular_matrix(self) -> Tex:
        q = MathTex(r"L", font_size=36)
        return q
    
    def _dense_upper_triangular_matrix(self) -> Tex:
        q = MathTex(r"U", font_size=36)
        return q
    
    def _dense_forward_substitution(self) -> Tex:
        pass

    def _dense_backward_substitution(self) -> Tex:
        pass

    def _sparsifying_dense_lower_upper_triangular_matrix(self) -> Tex:
        pass

    def _play_scene(self):
        # Linear system definition
        linear_sys_definition = self._linear_sys_definition()
        linear_sys_definition.center()

        #Create the dense matrix
        A_mat = self._create_dense_matrix()
        A_pattern = self._create_matrix_tex_pattern(A_mat.shape[0], A_mat.shape[1], A_mat)
        A_word = MathTex(r"A", font_size=36)
        A_word.next_to(A_pattern, UP)
        A_group = VGroup(A_pattern, A_word)

        # Multiplication of A and x
        mult_sign = MathTex(r"\times", font_size=36)
        #Create dense vector x in Ax=b
        x_pattern = self._create_dense_column_vector(A_mat.shape[0])
        x_word = MathTex(r"x", font_size=36)
        x_word.next_to(x_pattern, UP)
        x_group = VGroup(x_pattern, x_word)

        #Create the equal sign
        equal_sign = MathTex(r"=", font_size=36)

        #Create dense vector b in Ax=b
        b_pattern = self._create_dense_column_vector(A_mat.shape[0])
        b_word = MathTex(r"b", font_size=36)
        b_word.next_to(b_pattern, UP)
        b_group = VGroup(b_pattern, b_word)

        dense_eq = VGroup(A_group, mult_sign, x_group, equal_sign, b_group).arrange(RIGHT, buff=0.5)
        dense_eq.center()


        self.scene.play(Write(linear_sys_definition), run_time=self.transform_runtime)
        self.scene.wait(self.wait_time)

        self.scene.play(Transform(linear_sys_definition, dense_eq), run_time=self.transform_runtime)
        self.scene.wait(self.wait_time)


        A_sp = self._create_sparse_matrix(0, 0.1)
        A_sp_pattern = self._create_matrix_tex_pattern(A_sp.shape[0], A_sp.shape[1], A_sp)
        A_sp_word = MathTex(r"A", font_size=36)
        A_sp_word.next_to(A_sp_pattern, UP)
        A_sp_group = VGroup(A_sp_pattern, A_sp_word)
        
        dense_eq_new = VGroup(A_sp_group, mult_sign, x_group, equal_sign, b_group).arrange(RIGHT, buff=0.5)
        dense_eq_new.center()

        self.scene.play(Transform(dense_eq, dense_eq_new), run_time=self.transform_runtime)
        self.scene.wait(self.wait_time)


    def play_scene(self):
        # Entry point for Section 1 animation, with or without voiceover
        if isinstance(self.scene, VoiceoverScene):
            scripts = "Solving a symmetric semi-positive definite system of linear equations is a core numerical task in many applications. \
            For accurate computations of the solution 'X', a direct methods such Cholesky factorization is used. \
            The Cholesky factorization decompose the matrix into the multiplecation of a lower triangular matrix and its transpose. \
            Followed by a forward and backward substitution to solve for the solution 'X'. \
            Often time, in applications such as those using Finite Element methods, The matrix A is sparse.\
            In these applications, approximately more than 95\% of the entries are zero.\
            To exploit this sparsity, the matrix is stored in sparse formats, such as Coordinate list, or compressed row-format.\
            To have fast codes with these format, state-of-the-art sparse solvers employ two steps.\
            First, the sparsity pattern of matrix A is analyzed in the phase called symbolic analysis. Where depending on the sparsity \
            a set of acceleration techniques are selected and used for faster execution of numerical computations. \
            Second, the numerical computations are performed in the phase called numeric analysis. \
            While the symbolic analysis is often time more expensive than the executor, this two-step approach is justifiable, as in \
            these applications, often time the symbolic analysis is only performed once, and the numeric analysis is performed multiple times."
            with self.scene.voiceover(text=scripts) as tracker:    
                self._play_scene()
        else:
            self._play_scene()