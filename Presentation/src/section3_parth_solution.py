from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
from typing import Tuple
from utils import *

# 2) extend the pre-amble
template = TexTemplate()
template.add_to_preamble(r"\usepackage{xcolor}")

config.tex_template = template

# Custom DashedArcBetweenPoints class
class DashedArcBetweenPoints(DashedVMobject):
    def __init__(self, start, end, angle=-PI/2, radius=None, num_dashes=15, dashed_ratio=0.5, **kwargs):
        # Create the base arc
        arc = ArcBetweenPoints(start, end, angle=angle, radius=radius, **kwargs)
        
        # Initialize DashedVMobject with the arc
        super().__init__(arc, num_dashes=num_dashes, dashed_ratio=dashed_ratio, **kwargs)

# layout can still be your manual function, NX only supplies the signature
def custom_layout(g, scale=1):
    return {
        0:(0,0,0), 1:(0,1,0), 2:(1,1,0),
        3:(1,0,0), 4:(0,-1,0),5:(-1,-1,0),
        6:(-1,0,0),7:(-1,1,0), 8:(1,-1,0),
    }

class ParthToyExample(VGroup): 
    def __init__(self, matrix: np.ndarray, color_nodes: dict[int, str] = None, font_size: int = 24, **kwargs):
        super().__init__(**kwargs)
        n = matrix.shape[0]
        self.verts = list(range(n))
        # only include each undirected edge once
        self.edges = [(i, j) for i in range(n) for j in range(i+1, n) if matrix[i, j] != 0]

        self.default_style = {
            "stroke_color": BLACK,
            "stroke_width": 2,
        }


        self.vertex_config = {}
        if color_nodes is not None:
            for node in self.verts:
                if node in color_nodes:
                    self.vertex_config[node] = {"fill_color": color_nodes[node], "fill_opacity": 1, "stroke_color": BLACK, "stroke_width": 1}
                else:
                    self.vertex_config[node] = {"fill_color": GREEN_A, "fill_opacity": 1, "stroke_color": BLACK, "stroke_width": 1}
        else:
            self.vertex_config = {"fill_color": GREEN_A, "fill_opacity": 1, "stroke_color": BLACK, "stroke_width": 1}

        
        def ParthNode(label: int, fill_color: str = GREEN_A, fill_opacity: float = 1, stroke_color: str = BLACK, stroke_width: float = 1):
            """Create a pretty random star."""
            # show the block identifier as e.g. 𝔅₀, 𝔅₁, …
            label_text = MathTex(r"{%d}" % label, font_size=font_size, color=BLACK)
            surrounding_circle = Circle().surround(label_text, buff=0.2)
            surrounding_circle.set_fill(fill_color, fill_opacity=fill_opacity)
            surrounding_circle.set_stroke(stroke_color, stroke_width=stroke_width)
            return VGroup(surrounding_circle, label_text)
    
        self.G = Graph(
            self.verts,
            self.edges,
            labels={v for v in self.verts},
            vertex_type=ParthNode,
            vertex_config=self.vertex_config,
            layout=custom_layout,
            edge_config=self.default_style,
        )

        # Remove the straight line for (2,8)
        if (2, 8) in self.G.edges:
            self.G.remove_edges((2, 8))

            # # Optionally, add an updater so it follows the nodes:
            self.G.add_edges(
                (2, 8),
                edge_type=ArcBetweenPoints,
                edge_config={
                    "angle": -PI / 2,
                    "stroke_color": BLACK,
                    "stroke_width": 2,
                },
            )


        edges = VGroup(*self.G.edges.values())
        nodes = VGroup(*self.G.vertices.values())
        self.add(edges, nodes)

    def get_nodes(self, index: int)->Mobject:
        return self.G.vertices[index]
    
    def get_edges(self, edge: Tuple[int,int])->Mobject:
       return self.G.edges[edge]

    def remove_edges(self, edges: list[Tuple[int,int]]):
        for edge in edges:
            self.G.remove_edges(edge)

        self[0] = VGroup(*self.G.edges.values())

    def remove_nodes(self, nodes: list[int]):
        for node in nodes:
            self.G.remove_vertices(node)

        self[1] = VGroup(*self.G.vertices.values())

    def change_labels(self, new_labels: dict[int, int]):
        # Update the labels dictionary
        anims = []
        for node, new_label in new_labels.items():
                if node in self.labels.keys():
                    # Get current font size
                    node = self.get_nodes(node)
                    
                    prev_circle = node[0]
                    prev_label = node[1]
                    label_text = MathTex(r"{%d}" % new_label, font_size=prev_label.font_size, color=BLACK)
                    surrounding_circle = Circle().surround(label_text, buff=0.2)
                    anims.append(surrounding_circle.animate.set_fill(prev_circle.get_fill_color(), fill_opacity=prev_circle.get_fill_opacity()))
                    anims.append(surrounding_circle.animate.set_stroke(prev_circle.get_stroke_color(), stroke_width=prev_circle.get_stroke_width()))
                    anims.append(prev_label.animate.become(label_text))
                    anims.append(prev_circle.animate.become(surrounding_circle))
        return AnimationGroup(*anims, lag_ratio=0.05)


    def color_nodes(self, nodes: list[int], color: str):
        anims = []
        for node in nodes:
            node_vgroup = self.get_nodes(node)
            anims.append(node_vgroup[0].animate.set_fill(color))
        return AnimationGroup(*anims, lag_ratio=0.05)
      

    def make_edge_red(self, edge: Tuple[int,int]):
        # Remove the straight line for (2,8)
        anims = []
        if edge in self.G.edges:
            self.G.remove_edges(edge)

            # # Optionally, add an updater so it follows the nodes:
            anims.append(self.G.animate.add_edges(
                edge,
                edge_config={
                    "stroke_color": RED,
                    "stroke_width": 2,
                },
            ))
        return AnimationGroup(*anims, lag_ratio=0.05)

    def make_edge_dot_red(self, edge: Tuple[int,int]):
        # Remove the straight line for (2,8)
        anims = []
        if edge in self.G.edges:
            self.G.remove_edges(edge)

            # # Optionally, add an updater so it follows the nodes:
            anims.append(self.G.animate.add_edges(
                edge,
                edge_type=DashedArcBetweenPoints,
                edge_config={
                    "angle": -PI / 2,
                    "stroke_color": RED,
                    "stroke_width": 2,
                    "num_dashes": 10,
                    "dashed_ratio": 0.5,
                },
            ))
        return AnimationGroup(*anims, lag_ratio=0.05)
             

class HGDToyExample(VGroup):
    """
    This class is used to visualize the HGD decomposition of a matrix.
    It creates a binary tree
    """
    def __init__(self, level: int, color_nodes: dict[int, str] = None, font_size: int = 24, **kwargs):
        super().__init__(**kwargs)
        
        edges = []
        verts = list(range(2 ** (level + 1) - 1))
        for ancestor in range(2 ** level - 1):
            child_id = 2 * ancestor + 1
            child_id_2 = 2 * ancestor + 2   
            edges.append((child_id, ancestor))
            edges.append((child_id_2, ancestor))


        default_style = {
            "stroke_color": BLACK,
            "stroke_width": 2,
        }

    
        vertex_config = {}
        if color_nodes is not None:
            for node in verts:
                if node in color_nodes:
                    vertex_config[node] = {"fill_color": color_nodes[node], "fill_opacity": 1, "stroke_color": BLACK, "stroke_width": 1}
                else:
                    vertex_config[node] = {"fill_color": GREEN_A, "fill_opacity": 1, "stroke_color": BLACK, "stroke_width": 1}
        else:
            vertex_config = {"fill_color": GREEN_A, "fill_opacity": 1, "stroke_color": BLACK, "stroke_width": 1}

        def custom_layout(g, scale=2):
            return {
                0:(0,1,0), 1:(-1,0,0), 2:(1,0,0),
                3:(-2,-1,0), 4:(-1,-1,0),5:(1,-1,0),
                6:(2,-1,0)
            }
        
        def HGDNode(label: int, fill_color: str = GREEN_A, fill_opacity: float = 1, stroke_color: str = BLACK, stroke_width: float = 1):
            """Create a pretty random star."""
            # show the block identifier as e.g. 𝔅₀, 𝔅₁, …
            label_text = MathTex(r"\mathcal{B}_{%d}" % label, font_size=font_size, color=BLACK)
            rec = SurroundingRectangle(label_text, corner_radius=0.1, fill_color=fill_color,
                                    fill_opacity=fill_opacity, stroke_color=stroke_color, stroke_width=stroke_width)
            return VGroup(rec, label_text)
        
        self.G = Graph(
            verts,
            edges,
            labels={v: v for v in verts},
            vertex_config=vertex_config,
            vertex_type=HGDNode,
            layout=custom_layout,
            edge_config=default_style,
        )

        # Add the graph as the sole child of this VGroup so it can be accessed
        # later via ``self[0]`` (keeps the public API unchanged).
        nodes = VGroup(*self.G.vertices.values())
        edges = VGroup(*self.G.edges.values())
        self.add(edges, nodes)
        # Start with everything invisible – we will reveal pieces on demand via
        # ``add_nodes``.
        for mobj in [*nodes, *edges]:
            mobj.set_opacity(0)


    def get_nodes(self, index: int)->Mobject:
        return self.G.vertices[index]
    
    def get_edges(self, edge: Tuple[int,int])->Mobject:
       return self.G.edges[edge]
    
    def color_nodes(self, nodes: list[int], color: str):
        anims = []
        for node in nodes:
            node_vgroup = self.get_nodes(node)
            # node_vgroup[0] is the re
            # ctangle, node_vgroup[1] is the text
            rectangle = node_vgroup[0]
            text_label = node_vgroup[1]
            
            # Animate the rectangle fill color
            anims.append(rectangle.animate.set_fill(color))
            # Ensure text stays black and visible
            anims.append(text_label.animate.set_color(BLACK))
            
        return AnimationGroup(*anims, lag_ratio=0.05)

    def add_nodes(self, nodes: list[int], run_time: float = 1):
        """Reveal *nodes* (and their connecting edges) with a simple animation.

        Parameters
        ----------
        nodes : list[int]
            Indices (0-based) of tree vertices to reveal.
        run_time : float, optional
            Default duration of each *Create*/ *FadeIn* animation.  *Scene.play*
            may override this via its own ``run_time`` keyword.

        Returns
        -------
        AnimationGroup
            Can be passed straight to ``Scene.play``.
        """
        anims = []

        for node in nodes:
            # ----------------------------------------------------------------
            # 1) reveal the vertex & its label
            # ----------------------------------------------------------------
            vertex_mobj = self.get_nodes(node)
            vertex_mobj.set_opacity(1)
            anims.append(FadeIn(vertex_mobj, run_time=run_time))

            # ----------------------------------------------------------------
            # 2) reveal the edge towards the ancestor (except for the root)
            # ----------------------------------------------------------------
            if node != 0:  # root has no parent
                ancestor = (node - 1) // 2
                edge_key = (node, ancestor)
                edge_mobj = self.get_edges(edge_key)
                
                edge_mobj.set_opacity(1)
                anims.append(Create(edge_mobj, run_time=run_time))

        return AnimationGroup(*anims, lag_ratio=0.05)
    
    def reduce_opacity_of_edges(self, edges: list[Tuple[int,int]], run_time: float = 1):
        anims = []
        for edge in edges:
            edge_mobj = self.get_edges(edge)
            edge_mobj.set_opacity(0.2)
            anims.append(FadeIn(edge_mobj, run_time=run_time))
        return AnimationGroup(*anims, lag_ratio=0.05)
    

    def add_changed_edges(self):
        # Optionally, add an updater so it follows the nodes:
        self.G.add_edges(
            (2,6),
            edge_type=DashedArcBetweenPoints,
            edge_config={
                "angle": -PI / 2,
                "stroke_color": RED,
                "stroke_width": 2,
                "num_dashes": 10,
                "dashed_ratio": 0.5,
            },
        )

        self.G.add_edges(
            (0,1),
            edge_type = ArcBetweenPoints,
            edge_config={
                "angle": PI / 2,
                "stroke_color": RED,
                "stroke_width": 2,
            },
        )
        self.G.add_edges(
            (5,6),
            edge_type = ArcBetweenPoints,
            edge_config={
                "angle": PI / 2,
                "stroke_color": RED,
                "stroke_width": 2,
            },
        )

        new_edges = VGroup(*self.G.edges.values())
        new_nodes = VGroup(*self.G.vertices.values())
        print(new_edges)
        self[0] = new_edges
        self[1] = new_nodes

    def high_node(self, nodes: list[int]):
        anims = []
        for node in nodes:
            ancestor = (node - 1) // 2
            anims.append(FadeOut(self.get_nodes(node), run_time=self.transform_runtime))
            anims.append(FadeOut(self.get_edges((ancestor, node)), run_time=self.transform_runtime))
    
        return AnimationGroup(*anims, lag_ratio=0.05)
        

# Section2: Handles the animation and logic for the Parth solution section of the FastTrack video.
class ParthSolution():
    def __init__(self, scene: Scene | VoiceoverScene):
        self.scene = scene
        self.transform_runtime = 0.5
        self.wait_time = 1
    

    def _parth_intergration_block(self, font_size=14, container_width=5, container_height=4) -> VGroup:
        # Create a visual block showing Parth integration with different libraries
        # The block contains a brain SVG (Parth) and arrows to Accelerate, CHOLMOD, and MKL Pardiso libraries

        # Create the main container rectangle
        container = Rectangle(
            width=container_width, 
            height=container_height, 
            stroke_color=BLACK, 
            stroke_width=2, 
            fill_opacity=0.2,
            fill_color=BLACK
        )
        
        # Load the brain SVG and position it on the left
        border_margin_scale = 0.05
        brain = SVGMobject("Figures/ParthSol/adapt.svg", use_svg_cache=True)
        brain.scale_to_fit_width(container_width / 5)
        brain.move_to(container.get_left() + RIGHT * brain.get_width() / 2 + RIGHT * border_margin_scale * container_width)
        # Add a subtitle to the brain
        brain_subtitle = Text("Parth", font_size=font_size, color=WHITE)
        brain_subtitle.scale_to_fit_width(brain.get_width() * 0.9)
        brain_subtitle.move_to(brain.get_bottom() + DOWN * 0.5)
        
        # Create rectangles for the libraries
        rect_height = container_height / 4
        rect_width = container_width / 2.2
        accelerate_rect = Rectangle(
            width=rect_width, 
            height=rect_height, 
            stroke_color=WHITE, 
            stroke_width=1.5,
            fill_opacity=0.3,
            fill_color=GREEN
        )

        cholmod_rect = Rectangle(
            width=rect_width, 
            height=rect_height, 
            stroke_color=WHITE, 
            stroke_width=1.5,
            fill_opacity=0.3,
            fill_color=YELLOW
        )
        
        mkl_rect = Rectangle(
            width=rect_width, 
            height=rect_height, 
            stroke_color=WHITE, 
            stroke_width=1.5,
            fill_opacity=0.4,
            fill_color=RED
        )
        
        # Arrange the rectangles in a column on the right
        rectangles = VGroup(accelerate_rect, cholmod_rect, mkl_rect)
        rectangles.arrange(DOWN, buff=0.1 * rect_height)
        rectangles.move_to(container.get_right() + LEFT * rectangles.width / 2 + LEFT * border_margin_scale * container_width)

        # Add text labels to the rectangles
        accelerate_text = Text("Accelerate", font_size=font_size, color=WHITE)
        accelerate_text.scale_to_fit_width(rect_width * 0.9)
        accelerate_text.move_to(accelerate_rect.get_center())
        
        cholmod_text = Text("CHOLMOD", font_size=font_size, color=WHITE)
        cholmod_text.scale_to_fit_width(rect_width * 0.9)
        cholmod_text.move_to(cholmod_rect.get_center())
        
        mkl_text = Text("MKL Pardiso", font_size=font_size, color=WHITE)
        mkl_text.scale_to_fit_width(rect_width * 0.9)
        mkl_text.move_to(mkl_rect.get_center())
        
        # Create arrows from brain to each rectangle (horizontal arrows)
        arrow1 = Arrow(
            brain.get_right() + RIGHT * border_margin_scale * container_width,
            accelerate_rect.get_left() + LEFT * border_margin_scale * container_width,
            buff=0.1,
            stroke_width=2,
            color=WHITE
        )
        
        arrow2 = Arrow(
            brain.get_right() + RIGHT * border_margin_scale * container_width,
            cholmod_rect.get_left() + LEFT * border_margin_scale * container_width,
            buff=0.1,
            stroke_width=2,
            color=WHITE
        )
        
        arrow3 = Arrow(
            brain.get_right() + RIGHT * border_margin_scale * container_width,
            mkl_rect.get_left() + LEFT * border_margin_scale * container_width,
            buff=0.1,
            stroke_width=2,
            color=WHITE
        )
        
        # Group all elements together
        integration_block = VGroup(
            container,
            brain,
            brain_subtitle,
            accelerate_rect,
            cholmod_rect,
            mkl_rect,
            accelerate_text,
            cholmod_text,
            mkl_text,
            arrow1,
            arrow2,
            arrow3
        )
        
        return integration_block


    def _create_bar_chart(self)->CustomBarChart:
        init_vals   = [0.1, 0.1, 0.1]
        names       = ["MKL Pardiso", "Accelerate", "CHOLMOD"]

        chart = CustomBarChart(
            init_vals,                 # initial bar heights
            bar_names=names,
            y_range=[0, 6, 1],
            y_length=2,
            x_length=3,
            label_font_size=22,
        )
        return chart
    
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
    
    def _create_paper_sparse_matrix_after_change(self) -> np.ndarray:
        # Create the COO list
        row_indices = [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8]
        col_indices = [0, 6, 0, 1, 1, 2, 0, 2, 3, 8, 0, 4, 4, 5, 5, 6, 1, 6, 7, 4, 8]
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
    
    def _get_post_order_traversal(self)->VGroup:
        post_order = VGroup()
        post_order_idx = [3, 4, 1, 5, 6, 2, 0]
        Matrix_tex = []
        for i in range(len(post_order_idx)):
            Matrix_tex.append(r"\mathcal{B}_{%d}" % post_order_idx[i])
        post_order = Matrix([Matrix_tex], color=BLACK)
        post_order.get_brackets().set_color(BLACK)
        color_nodes={0: PURPLE, 1: GREEN, 2: YELLOW, 3: BLUE, 4: GOLD, 5: MAROON, 6: TEAL}
        post_order.get_brackets().set_color(BLACK)
        for i in range(len(post_order.get_entries())):
            post_order.get_entries()[i].set_color(color_nodes[post_order_idx[i]])
        label = Tex(r"Post-order:", color=BLACK)
        post_order = VGroup(label, post_order).arrange(RIGHT, buff=0.2)
        return post_order
    
    def _node_per_subgraph(self)->Matrix:
        node_per_subgraph_values = [1, 1, 1, 1, 1, 1, 3]
        node_per_subgraph = Matrix([node_per_subgraph_values], color=BLACK)
        node_per_subgraph.get_brackets().set_color(BLACK)
        return node_per_subgraph
    
    def _offset_per_subgraph(self)->VGroup:
        color_nodes={0: PURPLE, 1: GREEN, 2: YELLOW, 3: BLUE, 4: GOLD, 5: MAROON, 6: TEAL}
        post_order_idx = [3, 4, 1, 5, 6, 2, 0]
        offset_per_subgraph_values = [0, 1, 2, 3, 4, 5, 6, 9]
        offset_per_subgraph = Matrix([offset_per_subgraph_values])
        offset_per_subgraph.get_brackets().set_color(BLACK)
        for i in range(len(offset_per_subgraph.get_entries())):
            if i < 6:
                offset_per_subgraph.get_entries()[i].set_color(color_nodes[post_order_idx[i]])
            else:
                offset_per_subgraph.get_entries()[i].set_color(color_nodes[post_order_idx[6]])
        label = Tex(r"Offset:", color=BLACK)
        offset = VGroup(label, offset_per_subgraph).arrange(RIGHT, buff=0.2)
        return offset
    
    def _local_permutation_vectors(self)->Matrix:
        local_perm_values = [[0], [0], [0], [0], [0], [0], [0], [0], [1, 2, 0]]
        Matrices = []
        for i in range(len(local_perm_values)):
            Matrices.append(Matrix([local_perm_values[i]], color=BLACK))
        local_permutation_vectors = VGroup(*Matrices)
        local_permutation_vectors.arrange(RIGHT, buff=0.5)
        for i in range(len(local_permutation_vectors)):
            local_permutation_vectors[i].get_brackets().set_color(BLACK)
        return local_permutation_vectors
    
    def _global_permutation_vector(self)->Matrix:
        color_nodes={0: PURPLE, 1: GREEN, 2: YELLOW, 3: BLUE, 4: GOLD, 5: MAROON, 6: TEAL}
        post_order_idx = [3, 4, 1, 5, 6, 2, 0]
        global_perm_values = [7, 5, 6, 3, 8, 2, 1, 4, 0]
        global_permutation_vector = Matrix([global_perm_values])
        global_permutation_vector.get_brackets().set_color(BLACK)
        for i in range(len(global_permutation_vector.get_entries())):
            if i < 6:
                global_permutation_vector.get_entries()[i].set_color(color_nodes[post_order_idx[i]])
            else:
                global_permutation_vector.get_entries()[i].set_color(color_nodes[post_order_idx[6]])
        label = MathTex(r"P_G:", color=BLACK)
        global_permutation_vector = VGroup(label, global_permutation_vector).arrange(RIGHT, buff=0.2)
        return global_permutation_vector

    


    def play_parth_solution(self):
        get_center_of_section = self._get_centers_of_section(3)
        # Entry point for Section 2 animation, with or without voiceover
        if isinstance(self.scene, VoiceoverScene):
        #     script0 = "To accelerate fill-reducing ordering, we propose Parth which allows for reuse of the fill-reducing ordering\
        #         computation by integrating it into well-known sparse Cholesky solvers, Apple Accelerate, CHOLMOD, and MKL Pardiso."
        #     with self.scene.voiceover(text=script0) as tracker:
        #         # Integration block
        #         integration_block = self._parth_intergration_block().scale(0.8)
        #         brace_label = BraceLabel(integration_block, text=r"\text{Reliable Solution}", buff=0.1, font_size=32).set_color(BLACK)
        #         total_integration_block = VGroup(integration_block, brace_label)
        #         total_integration_block.move_to(get_center_of_section[0])
        #         self.scene.play(Create(total_integration_block), run_time=self.transform_runtime)
        #         self.scene.wait(self.wait_time)


        #     script1 = "With just 3 lines of code, and no tuning parameters, Parth adaptively provide high-quality fill-reducing ordering\
        #         in challenging applications where the sparsity pattern changes rapidly."
        #     with self.scene.voiceover(text=script1) as tracker:
        #         parth_code = Code(
        #             code_file="Materials/parth.cpp",
        #             tab_width=4,
        #             language="C++",
        #             background="rectangle",
        #             add_line_numbers=False,
        #             formatter_style="monokai",
        #         ).scale(0.6)
        #         brace_label = BraceLabel(parth_code, text=r"\text{Easy Integration}", buff=0.1, font_size=32).set_color(BLACK)
        #         total_parth_code = VGroup(parth_code, brace_label)
        #         total_parth_code.next_to(total_integration_block, RIGHT, buff=1)
        #         self.scene.play(Create(total_parth_code), run_time=self.transform_runtime)
        #         self.scene.wait(self.wait_time)

        #     script2 = "Its reuse capability allow for achieving up to 5.9× speedup per solves!"
        #     with self.scene.voiceover(text=script2) as tracker:
        #         speedup_chart = self._create_bar_chart()
        #         speedup_chart.next_to(total_parth_code, RIGHT, buff=1)
        #         brace_label = BraceLabel(speedup_chart, text=r"\text{High-Performance}", buff=0.1, font_size=32).set_color(BLACK)
        #         total_speedup_chart = VGroup(speedup_chart, brace_label)
        #         self.scene.play(Create(total_speedup_chart), run_time=self.transform_runtime)
        #         self.scene.wait(self.wait_time)
        #         self.scene.play(total_speedup_chart[0].animate_to_values([5.9, 3.8, 2.8], run_time=1))
        #         self.scene.wait(self.wait_time)

            
            # parth_steps_object = ParthStepsObject(step_description=["HGD", "Integrator", "Assembler"])
            # parth_steps_object.to_edge(UP)  
            # script3 = "It provides this performance benefits using 3 modules, namely,"
            # with self.scene.voiceover(text=script3) as tracker:
            #     #Fade out of the previous mobjects
            #     # self.scene.play(FadeOut(total_integration_block, total_parth_code, total_speedup_chart), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)
            #     #Create a the step object on the right
            #     self.scene.play(parth_steps_object.show_steps(), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)

            # script4 = "Hirerchical graph decomposition (or HGD in short)."
            # with self.scene.voiceover(text=script4) as tracker:
            #     #Fade out of the previous mobjects
            #     self.scene.play(parth_steps_object.show_steps([0]), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)

            
            # script5 = "Integrator."
            # with self.scene.voiceover(text=script5) as tracker:
            #     #Fade out of the previous mobjects
            #     self.scene.play(parth_steps_object.show_steps([0, 1]), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)

            
            # script6 = "Assembler."
            # with self.scene.voiceover(text=script6) as tracker:
            #     #Fade out of the previous mobjects  
            #     self.scene.play(parth_steps_object.show_steps([0, 1, 2]), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)

            # get_center_of_section = self._get_centers_of_section(2)
            # script7 = "In the first call to Parth, HGD algorithm get's the matrix A sparsity pattern."
            # with self.scene.voiceover(text=script7) as tracker:
            #     #Fade out of the previous mobjects  
            #     self.scene.play(parth_steps_object.highlight_step(0, highlight_color=RED), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)

            #     #Put the sparse matrix
            #     A_sp_initial = self._create_paper_initial_sparse_matrix()
            #     A_sp_pattern = create_matrix_tex_pattern(row_num=A_sp_initial.shape[0], col_num=A_sp_initial.shape[1], matrix=A_sp_initial, font_size=24)
            #     A_sp_pattern.move_to(get_center_of_section[0])
            #     self.scene.play(Create(A_sp_pattern), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)


            # script8 = "Then, same as other ordering algorithm, it look at matrix A as adjecent matrix of a graph."
            # with self.scene.voiceover(text=script8) as tracker:
            #     #Fade out of the previous mobjects  
            #     matrix_as_graph_initial = ParthToyExample(A_sp_initial, color_nodes={0: GREY_A, 1: GREY_A, 2: GREY_A, 3: GREY_A, 4: GREY_A, 5: GREY_A, 6: GREY_A, 7: GREY_A, 8: GREY_A})
            #     matrix_as_graph_initial.move_to(get_center_of_section[1])
            #     matrix_to_graph_arrow = Arrow(A_sp_pattern.get_right(), matrix_as_graph_initial.get_left(), buff=0.1, stroke_width=1, color=BLACK)
            #     self.scene.play(Create(matrix_to_graph_arrow), run_time=self.transform_runtime)
            #     self.scene.play(FadeIn(matrix_as_graph_initial), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)


            # HGD_example = HGDToyExample(level=2, color_nodes={0: PURPLE_A, 1: GREEN_A, 2: YELLOW_A, 3: BLUE_A, 4: GOLD_A, 5: MAROON_A, 6: TEAL_A})
            # HGD_example.move_to(get_center_of_section[1])
            # script9 = "Given the graph, Parth start decomposing the graph and build its tree representation of decomposition. Here, \
            #     the root of the tree with no children represent the whole graph."
            # with self.scene.voiceover(text=script9) as tracker:
            #     #Fade out of the previous mobjects  
            #     self.scene.play(FadeOut(A_sp_pattern), FadeOut(matrix_to_graph_arrow), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)
            #     #Move the graph to left
            #     self.scene.play(matrix_as_graph_initial.animate.move_to(get_center_of_section[0]), run_time=self.transform_runtime)
            #     #Show the HGD example
            #     new_matrix_graph_toy_example = ParthToyExample(A_sp_initial, color_nodes={0: PURPLE_A, 1: PURPLE_A, 2: PURPLE_A, 3: PURPLE_A, 4: PURPLE_A, 5: PURPLE_A, 6: PURPLE_A, 7: PURPLE_A, 8: PURPLE_A})
            #     new_matrix_graph_toy_example.move_to(matrix_as_graph_initial.get_center())
            #     self.scene.play(HGD_example.add_nodes([0]),
            #                     Transform(matrix_as_graph_initial, new_matrix_graph_toy_example),
            #                     run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)


            # script10 = "Parth then recursively decompose the graph into smaller subgraphs.\
            #     It first finds a separator set, which is a small number of nodes that if removed from the graph,\
            #         the graph will be split into two approximately equal-sized subgraphs. Here it is 1, 0 and 4. Note that\
            #         the B-zero is now updated to represent this new set."
            # with self.scene.voiceover(text=script10) as tracker:
            #     #Remove the edges 
            #     new_matrix_graph = new_matrix_graph_toy_example.copy()
            #     new_matrix_graph.remove_edges([(4, 5), (1, 7), (1,2), (0,3), (4,8)])
            #     self.scene.play(Transform(matrix_as_graph_initial, new_matrix_graph), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)


            # script10 = "After finding the set, the left and right subgraphs are now represent the left and right children of the root of HGD tree."
            # with self.scene.voiceover(text=script10) as tracker:
            #     #Fade out of the previous mobjects  
            #     new_matrix_graph.color_nodes([5, 6, 7], color=GREEN_A)
            #     new_matrix_graph.color_nodes([2, 3, 8], color=YELLOW_A)
                
            #     self.scene.play(Transform(matrix_as_graph_initial, new_matrix_graph), HGD_example.add_nodes([1, 2]), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)

            # # color_nodes={0: PURPLE_A, 1: GREEN_A, 2: YELLOW_A, 3: BLUE_A, 4: GOLD_A, 5: MAROON_A, 6: TEAL_A})
            # script11 = "This recursive process continues with the next leaves nodes in the tree,\
            # until a pre-determined level is reached., here the level is 2."
            # with self.scene.voiceover(text=script11) as tracker:
            #     #Fade out of the previous mobjects  
            #     new_matrix_graph.remove_edges([(2, 8), (2, 3), (6,7), (5,6)])
            #     new_matrix_graph.color_nodes([5], color=GOLD_A)
            #     new_matrix_graph.color_nodes([7], color=BLUE_A)
            #     new_matrix_graph.color_nodes([3], color=MAROON_A)
            #     new_matrix_graph.color_nodes([8], color=TEAL_A)
            #     self.scene.play(Transform(matrix_as_graph_initial, new_matrix_graph), HGD_example.add_nodes([3, 4, 5, 6]), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)

            # center0 = matrix_as_graph_initial.get_center()
            # center1 = HGD_example.get_center()
            # script12 = "Since we are in initialization phase, there is no change to be integrated, thus Parth moves to assembler step."
            # with self.scene.voiceover(text=script12) as tracker:
            #     #Fade out of the previous mobjects  
            #     self.scene.play(FadeOut(matrix_as_graph_initial), run_time=self.transform_runtime)
            #     self.scene.play(parth_steps_object.highlight_step(2, highlight_color=RED),
            #                     HGD_example.animate.move_to(center0), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)

            
            # script13 = "In the assembler phase, for each node in the HGD tree,\
            #     which represents a subgraph, Parth computes a local permutation vector and \
            #     then assembles them into the global permutation vector."
            # with self.scene.voiceover(text=script13) as tracker:
            #     pass

            

            # post_order_traversal = self._get_post_order_traversal()
            # post_order_traversal.scale_to_fit_width(5)

            # offset_per_subgraph = self._offset_per_subgraph()
            # offset_per_subgraph.scale_to_fit_width(post_order_traversal.get_width())

            # global_perm = self._global_permutation_vector()
            # global_perm.scale_to_fit_width(offset_per_subgraph.get_width() * 1.2)

            # positing = VGroup(post_order_traversal, offset_per_subgraph, global_perm)
            # positing.arrange(DOWN, aligned_edge=RIGHT, buff=0.5)
            # positing.to_edge(RIGHT, buff=1)

            
            # script14 = "To do that, it first finds the offset to identify the placement of each local permutation vector\
            #     based on the post-order traversal of the HGD tree representation.\ Here, this vector shows the post-order traversal of this tree."
            # with self.scene.voiceover(text=script14) as tracker:
            #     self.scene.play(Write(post_order_traversal), run_time=self.transform_runtime)

            # script15 = "Then using the number of nodes in each sub-graph representation,\
            #     it computes the offset for placing each local permutation vector withing global permutation vector\
            #         Note that the colored offset show the starting index of local permutation vector related to each sub-graph."
            # with self.scene.voiceover(text=script15) as tracker:
            #     self.scene.play(FadeOut(HGD_example), FadeIn(matrix_as_graph_initial), run_time=self.transform_runtime * 2)
            #     self.scene.wait(self.wait_time)
            #     self.scene.play(Write(offset_per_subgraph), run_time=self.transform_runtime)



            # script15 = "Then, it computes the local permutation vectors for each subgraph\
            #     and then integrate the local vectors into the global permutation vector.\
            #         Here, the final global permutation vector are shown. The color indicates where each local permutation vector is placed in the global permutation vector."
            # with self.scene.voiceover(text=script15) as tracker:
            #     self.scene.play(Write(global_perm), run_time=self.transform_runtime)


            # script15_1 = "To complete the example, Let's focus on computation of local permutation vector related to B-zero."
            # with self.scene.voiceover(text=script15_1) as tracker:
            #     local_graph = new_matrix_graph.copy()
            #     local_graph.remove_nodes([5, 6, 7, 2, 3, 8])
            #     local_perm_in_global_perm = global_perm[1].get_entries()[6:]
            #     local_perm_in_global_perm_rect = SurroundingRectangle(local_perm_in_global_perm, buff=0.1, color=PURPLE)
            #     self.scene.play(FadeOut(matrix_as_graph_initial), FadeIn(local_graph),
            #                     Create(local_perm_in_global_perm_rect), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)

            # script15_2 = "First, Parth re-name the B-zero sub-graph with local indices."
            # with self.scene.voiceover(text=script15_2) as tracker:
            #     renamed_local_graph = local_graph.copy()
            #     renamed_local_graph.change_labels({0: 0, 1: 1, 4: 2})
            #     self.scene.play(local_graph.animate.to_edge(LEFT, buff=1), run_time=self.transform_runtime)
            #     renamed_local_graph.next_to(local_graph, RIGHT, buff=1)
            #     arrow_graph = Arrow(local_graph.get_right(), renamed_local_graph.get_left(), color=PURPLE)
            #     self.scene.play(Create(arrow_graph), Create(renamed_local_graph), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)

            # local_perm = Matrix([[1, 2, 0]], color=PURPLE)
            # local_perm.get_brackets().set_color(BLACK)
            # local_perm.get_entries().set_color(PURPLE)
            # local_perm.scale_to_fit_width(1.5)
            # local_perm_in_global_perm = Matrix([[1, 4, 0]], color=PURPLE)
            # local_perm_in_global_perm.get_brackets().set_color(BLACK)
            # local_perm_in_global_perm.get_entries().set_color(PURPLE)
            # local_perm_in_global_perm.scale_to_fit_width(1.5)
            # local_order_group = VGroup(local_perm, local_perm_in_global_perm)
            # local_order_group.arrange(DOWN, aligned_edge=RIGHT, buff=1)
            # local_order_group.next_to(renamed_local_graph, RIGHT, buff=1)

            
            # script15_3 = "Then, a local permutation vector is computed, using fill-reducing ordering algorithms such as metis"
            # with self.scene.voiceover(text=script15_3) as tracker:
            #     self.scene.play(Create(local_perm), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)

            # script15_4 = "finally this local permutation vector is mapped into the global permutation vector,\
            #       by converting the node indices to their corresponding global names."
            # with self.scene.voiceover(text=script15_4) as tracker:
            #     arrow_perm = Arrow(local_perm.get_bottom(), local_perm_in_global_perm.get_top(), color=PURPLE)
            #     self.scene.play(Create(arrow_perm), Create(local_perm_in_global_perm), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)

            # script16 = "Once the assembler step is complete, the permutation vector is going to be reused until\
            #     a subsequent change in sparsity pattern occurs."
            # with self.scene.voiceover(text=script16) as tracker:
            #     pass

            # script17 = "When the sparsity pattern changes, Parth start the integrator step\
            #     to update the HGD tree representation."
            # with self.scene.voiceover(text=script17) as tracker:
            #     #fadeOut everything but parth_step_object
            #     objects_to_fade = [local_graph, renamed_local_graph, arrow_graph, arrow_perm,local_order_group, positing, local_perm_in_global_perm_rect]
            #     self.scene.play(FadeOut(*objects_to_fade), run_time=self.transform_runtime)
            #     self.scene.play(parth_steps_object.highlight_step(1, highlight_color=RED), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)

            script18 = "Here two non-zero entries are added and one is removed.\
                This changes the connectivity of the graph as shown."
            with self.scene.voiceover(text=script18) as tracker:
                old_matrix = self._create_paper_initial_sparse_matrix()
                new_matrix = self._create_paper_sparse_matrix_after_change()
                old_matrix_pattern = create_manim_Matrix(row_num=old_matrix.shape[0], col_num=old_matrix.shape[1], matrix=old_matrix)
                new_matrix_pattern = create_manim_Matrix(row_num=new_matrix.shape[0], col_num=new_matrix.shape[1], matrix=new_matrix)
                #make all the entries in old_matrix and new_matrix as 0 and 1
                old_matrix = np.where(old_matrix != 0, 1, 0)
                new_matrix = np.where(new_matrix != 0, 1, 0)
                diff = np.where(old_matrix != new_matrix, 1, 0)
                #find the entries that are 1 in diff
                diff_entries = np.where(diff == 1)    
                old_matrix_pattern.scale_to_fit_width(5)
                new_matrix_pattern.scale_to_fit_width(5)
                old_matrix_pattern.move_to(-3.5 * RIGHT)
                new_matrix_pattern.move_to(3.5 * RIGHT)
                old_matrix_pattern.get_brackets().set_color(BLACK)
                new_matrix_pattern.get_brackets().set_color(BLACK)
                self.scene.play(Create(old_matrix_pattern), Create(new_matrix_pattern), run_time=self.transform_runtime)
                rec_list = VGroup()
                for i in range(diff_entries[0].shape[0]):
                    row = diff_entries[0][i]
                    col = diff_entries[1][i] 
                    rect = SurroundingRectangle(new_matrix_pattern.get_entries()[row * new_matrix.shape[1] + col], buff=0.1, color=RED)
                    rect.set_stroke(width=2)
                    rec_list.add(rect)
                self.scene.play(Create(rec_list), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)
                self.scene.play(FadeOut(old_matrix_pattern, new_matrix_pattern), FadeOut(rec_list), run_time=self.transform_runtime)
                new_graph = ParthToyExample(new_matrix, color_nodes={0: GREY_A, 1: GREY_A, 2: GREY_A, 3: GREY_A, 4: GREY_A, 5: GREY_A, 6: GREY_A, 7: GREY_A, 8: GREY_A})
                new_graph.move_to(new_matrix_pattern.get_center())
                old_graph = ParthToyExample(old_matrix, color_nodes={0: GREY_A, 1: GREY_A, 2: GREY_A, 3: GREY_A, 4: GREY_A, 5: GREY_A, 6: GREY_A, 7: GREY_A, 8: GREY_A})
                old_graph.move_to(old_matrix_pattern.get_center())
                self.scene.play(FadeIn(new_graph, old_graph), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)


            script20 = "Parth detects these changes by comparing the new graph with the old one."
            with self.scene.voiceover(text=script20) as tracker:
                node_0_center = old_graph.G.vertices[0].get_center()
                self.scene.play(new_graph.animate.move_to(node_0_center), run_time=self.transform_runtime * 3)
                new_graph.make_edge_red((0,6))
                new_graph.make_edge_red((3,8))
                old_graph.make_edge_dot_red((2,8))
                self.scene.wait(self.wait_time)
                self.scene.remove(new_graph, old_graph)
                self.scene.add(new_graph, old_graph)
                color_nodes={0: PURPLE_A, 1: GREEN_A, 2: YELLOW_A, 3: BLUE_A, 4: GOLD_A, 5: MAROON_A, 6: TEAL_A}
                node_to_hmd_node = {0: 0, 1: 0, 2: 2, 3: 5, 4: 0, 5: 4, 6: 1, 7: 3, 8: 6}
                self.scene.wait(self.wait_time)
                self.scene.play(FadeOut(new_graph, old_graph), run_time=self.transform_runtime)

            script20_1 = "Then, it maps these changes into the HGD tree representation. Here, the added edges 0 to 6 in the graph is mapped to a change between subgraphs B0 to B1,\
            and the added and removed edges 3 to 8, and 2 to 8 are mapped to a change between subgraphs B5 to B6 and B2 to B6 respectively."
            with self.scene.voiceover(text=script20_1) as tracker:
                for node in node_to_hmd_node.keys():
                        new_graph.color_nodes([node], color_nodes[node_to_hmd_node[node]])
                HGD_example = HGDToyExample(level=2, color_nodes={0: PURPLE_A, 1: GREEN_A, 2: YELLOW_A, 3: BLUE_A, 4: GOLD_A, 5: MAROON_A, 6: TEAL_A})
                HGD_example.add_changed_edges()
                HGD_example.add_nodes([0, 1, 2, 3, 4, 5, 6])
                HGD_example.to_edge(RIGHT, buff=1)
                self.scene.play(FadeIn(new_graph, HGD_example), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

            script21 = "As can be seen, only the change between subgraph B5 to B6 violates the separator set condition. That is, \
                B2 no longer separates the B5 and B6 subgraphs. Thus, the integrator should resolve this conflict and integrate this new information into HGD."
            with self.scene.voiceover(text=script21) as tracker:
                self.scene.play(HGD_example.reduce_opacity_of_edges([(0,1), (2,6)], run_time=self.transform_runtime))
                pass
                

            script22 = "To do that, Parth first finds the coarse-grain subgraphs that encompasses the changes,\
                here subgraphs B2, B5, and B6 which encompass nodes 2, 3, and 8 respectively."
            with self.scene.voiceover(text=script22) as tracker:
                self.scene.play(HGD_example.color_nodes([2, 5, 6], color=RED_A),
                                HGD_example.color_nodes([0, 1, 3, 4], color=GREY_A),
                                run_time=self.transform_runtime)

                pass

            script23 = "Then, it recomputes the subgraphs to find valid separator set and updates the corresponding nodes in HGD representation."
            with self.scene.voiceover(text=script23) as tracker:
                pass

            script24 = "Then Parth moves to the assembler step and same as the previous time, it computes the local permutation vectors.\
                and updates global permutation vector. Here, only updating 3 entries, reusing the rest of 6 entries, acheiving 66% reuse."
            with self.scene.voiceover(text=script24) as tracker:
                pass            
             
        else:
            pass



