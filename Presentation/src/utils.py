from manim import *

def create_dense_matrix(size: int) -> np.ndarray:
    # Create a dense matrix pattern for visualization
    matrix_data = np.random.randn(size, size)
    return matrix_data

def create_sparse_matrix(size: int, iteration: int, sparsity: float) -> np.ndarray:
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

def create_lower_triangular_matrix(matrix: np.ndarray) -> np.ndarray:
    # Create a lower triangular dense matrix pattern for visualization
    lower_triangular_matrix = np.tril(matrix)
    return lower_triangular_matrix

def create_upper_triangular_matrix(matrix: np.ndarray) -> np.ndarray:
    # Create a upper triangular dense matrix pattern for visualization
    upper_triangular_matrix = np.triu(matrix)
    return upper_triangular_matrix
    

def create_matrix_tex_pattern(row_num: int, col_num: int, matrix: np.ndarray, text_color: str = BLACK, font_size: int = None) -> Tex:
    # Create the LaTeX matrix string for visualization
    # In the diagonal it has the number of row/col
    col_align = "|" + "c" * col_num + "|"
    matrix_str = r"$\begin{array}{" + col_align + r"}"
    for i in range(row_num):
        row_parts = []
        for j in range(col_num):
            if i == j and row_num == col_num:
                row_parts.append(f"{i}")
            else:
                if abs(matrix[i, j]) > 1e-10:  # Non-zero element
                    row_parts.append(r"\ast")
                else:  # Zero element
                    row_parts.append("")  # Empty string for zero elements
        matrix_str += " & ".join(row_parts)
        if i < row_num - 1:
            matrix_str += r" \\ "
    matrix_str += r"\end{array}$"
    
    
    # Create the matrix as LaTeX
    matrix_pattern = Tex(matrix_str, color=text_color, font_size=font_size)  # Reduced font size for better fit # Scale down for better visibility
    return matrix_pattern

def create_matrix_tex_with_values(row_num: int, col_num: int, matrix: np.ndarray, text_color: str = BLACK,
                                   font_size: int = None, simplify: bool = True) -> Tex:
    # Create the LaTeX matrix string for visualization
    # In the diagonal it has the number of row/col
    if simplify and col_num > 6:
        col_align = "|" + "c" * 7 + "|"
    else:
        col_align = "|" + "c" * col_num + "|"
    matrix_str = r"$\begin{array}{" + col_align + r"}"
    for i in range(row_num):
        if simplify and col_num > 6:
            row_parts = []
            for j in range(3):
                if i == j and row_num == col_num:
                    row_parts.append(f"{i+1}")
                else:
                    # values to string
                    # make the values to have one decimal point
                    # if the value is int
                    if matrix[i, j].is_integer():
                        if matrix[i, j] != 0:
                            row_parts.append(str(matrix[i, j]))
                        else:
                            row_parts.append("")
                    else:
                        s = f"{matrix[i, j]:.1f}"   # or "{:.1f}".format(x)
                        row_parts.append(str(s))
            
            row_parts.append(r"\dots")
            for j in range(col_num - 3, col_num):
                if i == j and row_num == col_num:
                    row_parts.append(f"{i+1}")
                else:
                    # values to string
                    # make the values to have one decimal point
                    # if the value is int
                    if matrix[i, j].is_integer():
                        if matrix[i, j] != 0:
                            row_parts.append(str(matrix[i, j]))
                        else:
                            row_parts.append("")
                    else:
                        s = f"{matrix[i, j]:.1f}"   # or "{:.1f}".format(x)
                        row_parts.append(str(s))
            matrix_str += " & ".join(row_parts)
            if i < row_num - 1:
                matrix_str += r" \\ "
        else:
            row_parts = []
            for j in range(col_num):
                if i == j and row_num == col_num:
                    row_parts.append(f"{i+1}")
                else:
                    # values to string
                    # make the values to have one decimal point
                    # if the value is int
                    if matrix[i, j].is_integer():
                        if matrix[i, j] != 0:
                            row_parts.append(str(matrix[i, j]))
                        else:
                            row_parts.append("")
                    else:
                        s = f"{matrix[i, j]:.1f}"   # or "{:.1f}".format(x)
                        row_parts.append(str(s))
            matrix_str += " & ".join(row_parts)
            if i < row_num - 1:
                matrix_str += r" \\ "
    matrix_str += r"\end{array}$"

    # Create the matrix as LaTeX
    matrix_pattern = Tex(matrix_str, color=text_color, font_size=font_size)  # Reduced font size for better fit# Scale down for better visibility

    return matrix_pattern


    
def create_dense_column_vector(matrix_size: int, with_values: bool = False, text_color: str = BLACK, font_size: int = None) -> Tex:
    # Create a dense column vector pattern for visualization
    column_vector = np.random.randn(matrix_size, 1)
    column_vector_pattern = None
    if with_values:
        column_vector_pattern = create_matrix_tex_with_values(matrix_size, 1, column_vector, font_size=font_size)
    else:
        column_vector_pattern = create_matrix_tex_pattern(matrix_size, 1, column_vector, font_size=font_size)
    return column_vector_pattern
    
def create_dense_row_vector(matrix_size: int, font_size: int = None) -> Tex:
    # Create a dense row vector pattern for visualization
    row_vector = np.random.randn(1, matrix_size)
    row_vector_pattern = create_matrix_tex_pattern(1, matrix_size, row_vector, font_size=font_size)
    return row_vector_pattern
    

class moduleBox(VGroup):
    def __init__(self, label_text: str, font_size: int = None, text_color: str = BLACK, stroke_color: str = BLACK,
                block_total_width: float = 1.0, block_total_height: float = 1.0, fill_color: str = YELLOW_A, corner_radius: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.box = RoundedRectangle(width=block_total_width, height=block_total_height,
                                stroke_color=stroke_color, stroke_width=1.5, fill_opacity=0.3, fill_color=fill_color, corner_radius=corner_radius)
        self.block_text = Text(label_text, font_size=font_size, color=text_color)
        # Fit the text inside the block
        # self.block_text.scale_to_fit_width(self.box.get_width() * 0.8)
        # self.box.stretch_to_fit_height(self.block_text.get_height()* 1.2)
        self.block_text.move_to(self.box.get_center())

        self.add(self.box, self.block_text)


class LabeledBox(VGroup):
    def __init__(
        self,
        mobject: Mobject,
        label_text: str,
        *,
        stroke_color = WHITE,
        stroke_width = 2,
        fill_color = GREY_E,        # <-- light, but still visible
        fill_opacity = 0.15,
        label_buff = 0.2,           # distance of label from the border (x and y)
        label_font_size = 28,
        corner_radius = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        # the main rectangle
        self.box = SurroundingRectangle(
            mobject,
            buff        = 0.25,      # space all around the contents
            color       = stroke_color,
            stroke_width= stroke_width,
            fill_color  = fill_color,
            fill_opacity= fill_opacity,
            corner_radius=corner_radius,
        )

        # the label
        self.label = (
            MathTex(rf"\text{{{label_text}}}", font_size=label_font_size, color=stroke_color)
               .move_to(self.box.get_corner(UL))
               .align_to(self.box, LEFT)
               .shift(label_buff * UP)
        )

        # make sure the rectangle is rendered **behind** the contents and label
        self.add(self.box, mobject, self.label)
    

class SymbolicNumericFramework(VGroup):
    def __init__(self, A_sp: np.ndarray, label_font_size: int = 32,
                 matrix_size: int = 10,
                 iteration: int = 0,
                 random_matrix_sparsity: float = 0.1,
                 generate_random_pattern: bool = False,
                 generate_random_values: bool = False,
                 value_font_size: int = 24,
                 symbolic_box_color: str = YELLOW_B,
                 numeric_box_color: str = GREEN_A,
                 text_color: str = BLACK,
                 arrow_stroke_width: int = 1,
                 arrow_color: str = BLACK,
                 matrix_name: str = "A",
                 rhs_name: str = "b",
                 unknown_name: str = "x",
                 corner_radius: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        
        # Use iteration as seed for randomness
        np.random.seed(iteration)
        self.matrix_size = matrix_size
        self.A_sp = A_sp
        self.label_font_size = label_font_size
        self.value_font_size = value_font_size
        self.symbolic_box_color = symbolic_box_color
        self.numeric_box_color = numeric_box_color
        self.text_color = text_color
        self.arrow_stroke_width = arrow_stroke_width
        self.arrow_color = arrow_color
        self.corner_radius = corner_radius
        self.matrix_name = matrix_name
        self.rhs_name = rhs_name
        self.unknown_name = unknown_name
        if generate_random_pattern:
            self.A_sp = create_sparse_matrix(self.matrix_size, iteration, random_matrix_sparsity)
        if generate_random_values:
            self._assigned_random_values()
        
        self.symbolic_numeric_framework = self._create_symbolic_numeric_framework_object()
        self.add(self.symbolic_numeric_framework)


    def _assigned_random_values(self):
        #iterate over non-zeros of A_sp and assigned them a random value
        for i in range(self.A_sp.shape[0]):
            for j in range(self.A_sp.shape[1]):
                if self.A_sp[i, j] != 0:
                    self.A_sp[i, j] = np.random.randn()

    def _create_symbolic_numeric_framework_object(self) -> VGroup:
        #Add input matrix
        A_sp_values = create_matrix_tex_with_values(self.A_sp.shape[0], self.A_sp.shape[1], self.A_sp, text_color=self.text_color, font_size=self.value_font_size, simplify=False)
        b = create_dense_column_vector(self.A_sp.shape[0], with_values=True, text_color=self.text_color, font_size=self.value_font_size)
        

        A_label = MathTex(self.matrix_name, color=self.text_color, font_size=self.label_font_size)
        b_label = MathTex(self.rhs_name, color=self.text_color, font_size=self.label_font_size)
        and_label = MathTex(r"\text{,}", color=self.text_color, font_size=self.label_font_size)
        and_label.next_to(A_sp_values, RIGHT, buff=0.2)
        b.next_to(and_label, RIGHT, buff=0.2)
        A_label.next_to(A_sp_values, UP)
        b_label.next_to(b, UP)
        numeric_input_group = VGroup(A_sp_values, and_label, b, A_label, b_label).scale(0.8)

        x = create_dense_column_vector(self.A_sp.shape[0], with_values=True, font_size=self.value_font_size)
        x_label = MathTex(self.unknown_name, color=self.text_color, font_size=self.label_font_size)
        x_label.next_to(x, UP)
        solve_values = VGroup(x, x_label)

        A_sp_pattern = create_matrix_tex_pattern(self.A_sp.shape[0], self.A_sp.shape[1], self.A_sp, font_size=self.value_font_size)
        A_pattern_label = Text(self.matrix_name + "'s pattern", color=self.text_color, font_size=self.label_font_size)
        A_pattern_label.next_to(A_sp_pattern, UP)
        A_pattern_group = VGroup(A_sp_pattern, A_pattern_label)
        
        
        # Sparse Cholesky Sove
        symbolic_box = moduleBox(label_text="Symbolic", font_size=self.label_font_size, fill_color=self.symbolic_box_color, block_total_width=2, block_total_height=1.0, corner_radius=self.corner_radius)
        numeric_box = moduleBox(label_text="Numeric", font_size=self.label_font_size, fill_color=self.numeric_box_color, block_total_width=2, block_total_height=1.0, corner_radius=self.corner_radius)
        symbolic_box.next_to(A_sp_pattern, RIGHT, buff=1)
        numeric_box.next_to(symbolic_box, RIGHT, buff=1)
        sym_to_num_arrow = Arrow(symbolic_box.get_right(), numeric_box.get_left(), stroke_width=self.arrow_stroke_width, color=self.arrow_color)
        sparse_cholesky_framework = VGroup(symbolic_box, numeric_box, sym_to_num_arrow)
        sparse_cholesky_label = BraceLabel(sparse_cholesky_framework, text=r"\text{Sparse Cholesky Solver}", buff=0.1, font_size=self.label_font_size).set_color(self.arrow_color)
        numeric_input_group.next_to(numeric_box, UP, buff=1.3)
        solve_values.next_to(numeric_box, RIGHT, buff=1)

        arrow_pattern_to_symbolic = Arrow(A_sp_pattern.get_right(), sparse_cholesky_framework[0].get_left(), stroke_width=self.arrow_stroke_width, color=self.arrow_color)
        arrow_solve_input_to_numeric = Arrow(numeric_input_group.get_bottom(), sparse_cholesky_framework[1].get_top(), stroke_width=self.arrow_stroke_width, color=self.arrow_color)
        arrow_sparse_cholesky_to_solve_values = Arrow(numeric_box.get_right(), solve_values.get_left(), stroke_width=self.arrow_stroke_width, color=self.arrow_color)

        solve_example = VGroup(A_pattern_group, arrow_pattern_to_symbolic, sparse_cholesky_framework, sparse_cholesky_label,
                            numeric_input_group, arrow_solve_input_to_numeric, arrow_sparse_cholesky_to_solve_values, solve_values)
        return solve_example
    

class CustomBarChart(VGroup):
    """
    A BarChart that keeps its numeric labels (“1.2 ×”) and bars in sync.

    Parameters
    ----------
    values : list[float]
        Initial bar heights.
    bar_names : list[str]
        Strings shown on the x‑axis.
    y_range : list[float]
        [y_min, y_max, step] exactly as in manim.BarChart.
    y_length, x_length : float
        Chart dimensions in world‑units.
    scale_symbol : str, optional
        String to place after each DecimalNumber (default “×”).
    x_axis_config, y_axis_config : dict | None
        Passed straight to BarChart.
    label_font_size : int, optional
        Font size of DecimalNumbers and “×”.
    """
    def __init__(
        self,
        values: list[float],
        bar_names: list[str],
        y_range: list[float],
        y_length: float,
        x_length: float,
        *,
        scale_symbol: str = r"$\times$",
        x_axis_config: dict | None = None,
        y_axis_config: dict | None = None,
        label_font_size: int = 24,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # --- store arguments -------------------------------------------------
        self.current_values = list(values)            # local copy we mutate
        self.scale_symbol = scale_symbol
        self.label_font_size = label_font_size

        # sensible defaults
        # x_axis_config = x_axis_config or {"font_size": label_font_size}
        # y_axis_config = y_axis_config or {"font_size": label_font_size}

        # --------------------------------------------------------------------
        # 1) the underlying BarChart
        # --------------------------------------------------------------------
        self.chart = BarChart(
            values=values,
            bar_names=bar_names,
            y_range=y_range,
            y_length=y_length,
            x_length=x_length,
            x_axis_config={"font_size": label_font_size, "color": BLACK},
            y_axis_config={"font_size": label_font_size, "color": BLACK},
        )
        # rotate x‑axis labels for readability
        for lbl in self.chart.x_axis.labels:
            lbl.rotate(PI / 4).shift(DOWN * 0.2 + LEFT * 0.1)
            lbl.set_color(BLACK)
        
        for num in self.chart.y_axis.numbers:
            num.set_color(BLACK)

        # --------------------------------------------------------------------
        # 2) the read‑out (“1.2 ×”) for every bar
        # --------------------------------------------------------------------
        self.decimals = VGroup()    # DecimalNumber objects
        self.symbols  = VGroup()    # Tex objects with “×”

        for v in values:
            d = DecimalNumber(v, num_decimal_places=1, font_size=label_font_size, color=BLACK)
            s = Tex(self.scale_symbol, font_size=label_font_size, color=BLACK)
            self.decimals.add(d)
            self.symbols.add(s)

        # initially position them above their bars
        self._reposition_labels()

        # --------------------------------------------------------------------
        # 3) updaters — keep everything glued together each frame
        # --------------------------------------------------------------------
        def bar_updater(_):          # _ is the BarChart itself
            # update bar heights
            #Get values from decimal numbers
            values = [d.get_value() for d in self.decimals]
            self.chart.change_bar_values(values)
            # and re‑position labels
            self._reposition_labels()

        self.chart.add_updater(bar_updater)

        # --------------------------------------------------------------------
        # 4) assemble
        # --------------------------------------------------------------------
        self.add(self.chart, self.decimals, self.symbols)

        # --------------------------------------------------------------------
        # put the thin stuff *on top* of the thick rectangles
        # --------------------------------------------------------------------
        self.chart.bars.set_z_index(0)        # draw bars first
        self.chart.x_axis.set_z_index(1)
        self.chart.y_axis.set_z_index(1)
        self.decimals.set_z_index(2)
        self.symbols.set_z_index(2)

    # ------------------------------------------------------------------------
    # public helpers
    # ------------------------------------------------------------------------
    def set_values_instant(self, new_values: list[float]) -> None:
        """Change values immediately (no animation)."""
        self.current_values[:] = new_values
        for d, v in zip(self.decimals, new_values):
            d.set_value(v)

    def animate_to_values(self, new_values: list[float], run_time: float = 1):
        """
        Return an AnimationGroup that:
            • animates the DecimalNumbers to *new_values*
            • (the chart heights follow automatically via the updater)
        This is convenient so callers only need one play() line.
        """
        # keep reference so updater sees the target numbers evolving
        self.current_values[:] = new_values

        animations = [
            ChangeDecimalToValue(d, v) for d, v in zip(self.decimals, new_values)
        ]
        return AnimationGroup(*animations, run_time=run_time)

    # ------------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------------
    def _reposition_labels(self) -> None:
        """Place each decimal and its '×' just above its bar."""
        for idx, (d, s) in enumerate(zip(self.decimals, self.symbols)):
            d.next_to(self.chart.bars[idx], UP, buff=0.1)
            s.next_to(d, RIGHT, buff=0.05)


class ParthStepsObject(VGroup):
    def __init__(
        self,
        step_description: list[str],
        steps_sourrunding_color: str = GREEN_A,
        step_sorrunding_buff: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # ------------------------------------------------------------------ #
        # Build steps                                                         #
        # ------------------------------------------------------------------ #
        self.step_description = step_description
        self.step_mobjects   = len(step_description)
        self._visible        = set(range(self.step_mobjects))  # track state

        text_objects = []
        max_w = max_h = 0
        for txt in step_description:
            t = Text(txt, font_size=24, color=BLACK)
            text_objects.append(t)
            max_w = max(max_w, t.width)
            max_h = max(max_h, t.height)

        max_w += step_sorrunding_buff * 2
        max_h += step_sorrunding_buff * 2

        self.default_stroke_color = GREEN_A
        self.default_stroke_width = 0.5
        self.step_objects = VGroup()
        for t in text_objects:
            rect = RoundedRectangle(
                width=max_w, height=max_h,
                fill_color=self.default_stroke_color, fill_opacity=0.1,
                stroke_color=steps_sourrunding_color,
                stroke_width=self.default_stroke_width, corner_radius=0.1,
            )
            t.move_to(rect.get_center())
            self.step_objects.add(VGroup(rect, t))

        self.step_objects.arrange(RIGHT, buff=0.5)

        # outer frame
        self.parth_box = VGroup()

        self.highlight_step_number = None
        self.add(self.parth_box, *self.step_objects)

    # ────────────────────────────────────────────────────────────────────── #
    # Helpers                                                               #
    # ────────────────────────────────────────────────────────────────────── #
    def _get_step_mobjects(self, idx: int) -> VGroup:
        if not 0 <= idx < self.step_mobjects:
            raise ValueError(f"Index {idx} out of range.")
        return self[idx + 1]

    def all_steps(self) -> VGroup:
        """Return VGroup with *all* steps (handy for Indicate, etc.)."""
        return self.step_objects

    # ────────────────────────────────────────────────────────────────────── #
    # Animation‑based show_steps                                            #
    # ────────────────────────────────────────────────────────────────────── #
    def show_steps(
        self,
        step_numbers: list[int] | None = None,
        reveal_anim=Create,
        hide_anim=FadeOut,
        lag_ratio: float = 0.05,
        resize_box: bool = True,
    ) -> AnimationGroup:
        """
        Return an AnimationGroup that leaves only *step_numbers* visible.

        Parameters
        ----------
        step_numbers  – list of **0‑based** indices to reveal;
                        ``None`` ⇒ show **all** steps.
        reveal_anim / hide_anim – animation classes for appearing/disappearing.
        lag_ratio     – passed to the AnimationGroup for a slight cascade.
        resize_box    – if True, the outer rectangle shrinks/grows to fit
                        the *currently* visible steps.

        Usage example
        -------------
            parth = ParthStepsObject(["Load", "Clean", "Train", "Eval"])
            self.play(FadeIn(parth))                 # everything shows
            self.play(parth.show_steps([1, 2]))      # keep only Clean/Train
            self.wait()
            self.play(parth.show_steps([2]))         # now only Train
        """
        # ----------------------------- sanity ----------------------------- #
        if step_numbers is None:
            new_visible = set()
        else:
            if any(not 0 <= i < self.step_mobjects for i in step_numbers):
                raise ValueError(
                    f"indices must be in 0..{self.step_mobjects-1}, "
                    f"got {sorted(step_numbers)}"
                )
            new_visible = set(step_numbers)

        old_visible = self._visible
        if new_visible == old_visible:
            return AnimationGroup()  # nothing to do

        # ----------------------------- diff ------------------------------- #
        to_show = new_visible - old_visible
        to_hide = old_visible - new_visible

        anims = []
        for idx in to_show:
            grp = self._get_step_mobjects(idx)
            grp.set_opacity(1)               # start hidden
            anims.append(reveal_anim(grp))

        for idx in to_hide:
            grp = self._get_step_mobjects(idx)
            grp.set_opacity(0)
            anims.append(hide_anim(grp))

        # --------------------------- resize box --------------------------- #
        if resize_box:
            if new_visible:
                tgt = SurroundingRectangle(
                    VGroup(*[self._get_step_mobjects(i) for i in new_visible]),
                    fill_color=BLUE_A, stroke_color=BLUE_B,
                    stroke_width=1, corner_radius=0.1, buff=0.1,
                )
                anims.insert(0, Transform(self.parth_box, tgt))
            else:  # no steps → fade the box out
                anims.insert(0, FadeOut(self.parth_box))

        # record new state
        self._visible = new_visible
        return AnimationGroup(*anims, lag_ratio=lag_ratio)

    # ────────────────────────────────────────────────────────────────────── #
    # Optional animated highlighter                                         #
    # ────────────────────────────────────────────────────────────────────── #
    def highlight_step(
        self,
        step_number: int,
        highlight_color: str | ManimColor = YELLOW,
        run_time: float = 0.5,
    ) -> AnimationGroup:
        """
        Animated tint of a single step’s stroke & text.

        Example
        -------
            self.play(parth.highlight_step(2))
        """
        grp = self._get_step_mobjects(step_number)
        rect, txt = grp
        if self.highlight_step_number is not None:
            prev_highlighted_step = self.highlight_step_number
            prev_grp = self._get_step_mobjects(prev_highlighted_step)
            prev_rect, prev_txt = prev_grp
            self.highlight_step_number = step_number
            return AnimationGroup(
                prev_rect.animate.set_stroke(color=self.default_stroke_color, width=self.default_stroke_width),
                prev_txt.animate.set_color(BLACK),
                rect.animate.set_stroke(color=highlight_color, width=2),
                txt.animate.set_color(highlight_color),
                lag_ratio=0.0, run_time=run_time,
            )
        else:
            self.highlight_step_number = step_number
            return AnimationGroup(
                rect.animate.set_stroke(color=highlight_color, width=2),
                txt.animate.set_color(highlight_color),
                lag_ratio=0.0, run_time=run_time,
            )



