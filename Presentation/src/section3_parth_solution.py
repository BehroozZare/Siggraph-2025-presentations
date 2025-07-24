from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
from utils import *

# 2) extend the pre-amble
template = TexTemplate()
template.add_to_preamble(r"\usepackage{xcolor}")

config.tex_template = template



class ParthToyExample(VGroup):
    def __init__(self, matrix: np.ndarray, color_nodes: dict[int, str] = None, font_size: int = 24, **kwargs):
        super().__init__(**kwargs)
        n = matrix.shape[0]
        verts = list(range(n))
        # only include each undirected edge once
        edges = [(i, j) for i in range(n) for j in range(i+1, n) if matrix[i, j] != 0]

        default_style = {
            "stroke_color": BLACK,
            "stroke_width": 2,
        }
        # layout can still be your manual function, NX only supplies the signature
        def custom_layout(g, scale=1):
            return {
                0:(0,0,0), 1:(0,1,0), 2:(1,1,0),
                3:(1,0,0), 4:(0,-1,0),5:(-1,-1,0),
                6:(-1,0,0),7:(-1,1,0), 8:(1,-1,0),
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

        self.G = Graph(
            verts,
            edges,
            labels={v: Text(str(v), font_size=font_size, color=BLACK) for v in verts},
            vertex_config=vertex_config,
            layout=custom_layout,
            edge_config=default_style,
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
        labels = VGroup(*self.G._labels.values())
        self.add(edges, nodes, labels)


class HGDToyExample(VGroup):
    """
    This class is used to visualize the HGD decomposition of a matrix.
    It creates a binary tree
    """
    def __init__(self, level: int, color_nodes: dict[int, str] = None, font_size: int = 24, **kwargs):
        super().__init__(**kwargs)
        
        edges = []
        verts = list(range(2 ** (level + 1) - 1))
        for ancestor in range(2 ** level):
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
        
        self.G = Graph(
            verts,
            edges,
            labels={v: Text(str(v), font_size=font_size, color=BLACK) for v in verts},
            vertex_config=vertex_config,
            layout=custom_layout,
            edge_config=default_style,
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

    def add_nodes(self, nodes: list[int], run_time: float = 1)->AnimationGroup:
        # For each node, add two childs to the node and increase the scale of the graph
        for node in nodes:
            self.add(self.G.vertices[node])
            ancestor = node // 2
            #for each added nodes, add the edges too
            self.add(self.G.edges[(node, ancestor)])

        return AnimationGroup(
            *[Create(self.G.edges[(node, ancestor)]) for node in nodes],
            *[Create(self.G.vertices[node]) for node in nodes],

            lag_ratio=0.0,
            run_time=1
        )
        
        
        

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
    


    def play_parth_solution(self):
        get_center_of_section = self._get_centers_of_section(3)
        # Entry point for Section 2 animation, with or without voiceover
        if isinstance(self.scene, VoiceoverScene):
            # script0 = "To accelerate fill-reducing ordering, we propose Parth which allows for reuse of the fill-reducing ordering\
            #     computation by integrating it into well-known sparse Cholesky solvers, Apple Accelerate, CHOLMOD, and MKL Pardiso."
            # with self.scene.voiceover(text=script0) as tracker:
            #     # Integration block
            #     integration_block = self._parth_intergration_block().scale(0.8)
            #     brace_label = BraceLabel(integration_block, text=r"\text{Reliable Solution}", buff=0.1, font_size=32).set_color(BLACK)
            #     total_integration_block = VGroup(integration_block, brace_label)
            #     total_integration_block.move_to(get_center_of_section[0])
            #     self.scene.play(Create(total_integration_block), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)


            # script1 = "With just 3 lines of code, and no tuning parameters, Parth adaptively provide high-quality fill-reducing ordering\
            #     in challenging applications where the sparsity pattern changes rapidly."
            # with self.scene.voiceover(text=script1) as tracker:
            #     parth_code = Code(
            #         code_file="Materials/parth.cpp",
            #         tab_width=4,
            #         language="C++",
            #         background="rectangle",
            #         add_line_numbers=False,
            #         formatter_style="monokai",
            #     ).scale(0.6)
            #     brace_label = BraceLabel(parth_code, text=r"\text{Easy Integration}", buff=0.1, font_size=32).set_color(BLACK)
            #     total_parth_code = VGroup(parth_code, brace_label)
            #     total_parth_code.next_to(total_integration_block, RIGHT, buff=1)
            #     self.scene.play(Create(total_parth_code), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)

            # script2 = "Its reuse capability allow for achieving up to 5.9× speedup per solves!"
            # with self.scene.voiceover(text=script2) as tracker:
            #     speedup_chart = self._create_bar_chart()
            #     speedup_chart.next_to(total_parth_code, RIGHT, buff=1)
            #     brace_label = BraceLabel(speedup_chart, text=r"\text{High-Performance}", buff=0.1, font_size=32).set_color(BLACK)
            #     total_speedup_chart = VGroup(speedup_chart, brace_label)
            #     self.scene.play(Create(total_speedup_chart), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)
            #     self.scene.play(total_speedup_chart[0].animate_to_values([5.9, 3.8, 2.8], run_time=1))
            #     self.scene.wait(self.wait_time)

            
            parth_steps_object = ParthStepsObject(step_description=["HGD", "Integrator", "Assembler"])
            parth_steps_object.to_edge(UP)
            script3 = "It provides this performance benefits using 3 modules, namely,"
            with self.scene.voiceover(text=script3) as tracker:
                #Fade out of the previous mobjects
                # self.scene.play(FadeOut(total_integration_block, total_parth_code, total_speedup_chart), run_time=self.transform_runtime)
                # self.scene.wait(self.wait_time)
                #Create a the step object on the right
                self.scene.play(parth_steps_object.show_steps(), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

            script4 = "Hirerchical graph decomposition (or HGD in short)."
            with self.scene.voiceover(text=script4) as tracker:
                #Fade out of the previous mobjects
                self.scene.play(parth_steps_object.show_steps([0]), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

            
            script5 = "Integrator."
            with self.scene.voiceover(text=script5) as tracker:
                #Fade out of the previous mobjects
                self.scene.play(parth_steps_object.show_steps([0, 1]), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

            
            script6 = "Assembler."
            with self.scene.voiceover(text=script6) as tracker:
                #Fade out of the previous mobjects  
                self.scene.play(parth_steps_object.show_steps([0, 1, 2]), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

            hgd_example = HGDToyExample(level=2)
            hgd_example.to_edge(LEFT, buff=2)


            get_center_of_section = self._get_centers_of_section(2)
            script7 = "In the first call to Parth, HGD algorithm get's the matrix A sparsity pattern."
            with self.scene.voiceover(text=script7) as tracker:
                #Fade out of the previous mobjects  
                self.scene.play(parth_steps_object.highlight_step(0, highlight_color=RED), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)

                #Put the sparse matrix
                A_sp_initial = self._create_paper_initial_sparse_matrix()
                A_sp_pattern = create_matrix_tex_pattern(row_num=A_sp_initial.shape[0], col_num=A_sp_initial.shape[1], matrix=A_sp_initial, font_size=24)
                A_sp_pattern.move_to(get_center_of_section[0])
                self.scene.play(Create(A_sp_pattern), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)


            script8 = "Then, same as other ordering algorithm, it look at matrix A as adjecent matrix of a graph."
            with self.scene.voiceover(text=script8) as tracker:
                #Fade out of the previous mobjects  
                matrix_as_graph_initial = ParthToyExample(A_sp_initial)
                matrix_as_graph_initial.move_to(get_center_of_section[1])
                matrix_to_graph_arrow = Arrow(A_sp_pattern.get_right(), matrix_as_graph_initial.get_left(), buff=0.1, stroke_width=1, color=BLACK)
                self.scene.play(Create(matrix_to_graph_arrow), run_time=self.transform_runtime)
                self.scene.play(FadeIn(matrix_as_graph_initial), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)


            HGD_example = HGDToyExample(level=2)
            script9 = "Given the graph, Parth start decomposing the graph and build its tree representation of decomposition. Here, \
                the root of the tree with no children represent the whole graph."
            with self.scene.voiceover(text=script8) as tracker:
                #Fade out of the previous mobjects  
                
                matrix_as_graph_initial = ParthToyExample(A_sp_initial)
                matrix_as_graph_initial.move_to(get_center_of_section[1])
                matrix_to_graph_arrow = Arrow(A_sp_pattern.get_right(), matrix_as_graph_initial.get_left(), buff=0.1, stroke_width=1, color=BLACK)
                self.scene.play(Create(matrix_to_graph_arrow), run_time=self.transform_runtime)
                self.scene.play(FadeIn(matrix_as_graph_initial), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)


            script9 = "Then here is the next call"
            with self.scene.voiceover(text=script9) as tracker:
                #Fade out of the previous mobjects  
                A_sp_after_change = self._create_paper_sparse_matrix_after_change()
                A_sp_after_change_pattern = create_matrix_tex_pattern(row_num=A_sp_after_change.shape[0], col_num=A_sp_after_change.shape[1], matrix=A_sp_after_change, font_size=24)
                A_sp_after_change_pattern.to_edge(LEFT, buff=2)
                self.scene.play(FadeOut(matrix_as_graph_initial), FadeIn(A_sp_after_change_pattern), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)


            script10 = "and here is the graph of the next call"
            with self.scene.voiceover(text=script10) as tracker:
                #Fade out of the previous mobjects  
                matrix_as_graph_after_change = ParthToyExample(A_sp_after_change)
                matrix_as_graph_after_change.move_to(A_sp_after_change_pattern.get_center())
                self.scene.play(FadeOut(A_sp_after_change_pattern), FadeIn(matrix_as_graph_after_change), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)


            script11 = "and here is coloring the nodes"
            with self.scene.voiceover(text=script11) as tracker:
                #Fade out of the previous mobjects  
                new_colored_matrix = ParthToyExample(A_sp_after_change, color_nodes={0: RED, 1: RED, 2: GREEN_A, 3: GREEN_A, 4: BLUE_A, 5: BLUE_A, 6: BLUE_A, 7: RED, 8: RED})
                self.scene.play(Transform(matrix_as_graph_after_change, new_colored_matrix), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)


                





            
            
            
            
            # script4 = "In first step,\
            #       it hirerchically, decomposes the graph dual of the matrix into multiple subgraphs as form \
            #         a tree representation of this decomposition."
            # with self.scene.voiceover(text=script3) as tracker:
            #     #Fade out of the previous mobjects
            #     self.scene.play(FadeOut(total_integration_block, total_parth_code, total_speedup_chart), run_time=self.transform_runtime)
            #     self.scene.wait(self.wait_time)
                #Create a the step object on the right


            # script4 = "This tree representation provide both fine-grain region as well as coarser-region, allowing for encompassing changes in these regions."
            # with self.scene.voiceover(text=script4) as tracker:
            #     pass
            
            # script5 = "These regions also have one to one mapping to local sub-vectors in the fill-reducing vector enabling the local update of this permutation vector."
            # with self.scene.voiceover(text=script5) as tracker:
            #     pass

            # script6 = "The second step, integrator, involves detecting and integrating the new changes into the tree representation and marking the sub-graphs that needs new local permutation vectors."
            # with self.scene.voiceover(text=script6) as tracker:
            #     pass

            # script7 = "And finally, the third step, assember, updates the local permutation \
            #     vectors, and integrate them into global permutation vector."
            # with self.scene.voiceover(text=script7) as tracker:
            #     pass
 
        else:
            pass
