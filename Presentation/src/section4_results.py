from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import numpy as np
import os
from slides.SCENE_CONFIG import *
import re

# 2) extend the pre-amble
template = TexTemplate()
template.add_to_preamble(r"\usepackage{xcolor}")
template.add_to_preamble(r"\usepackage{helvet}")  # For Helvetica font
template.add_to_preamble(r"\renewcommand{\familydefault}{\sfdefault}")  # Set sans-serif as default

config.tex_template = template


# Section1: Handles the animation and logic for the problem definition section of the FastTrack video.
class Results():
    def __init__(self, scene: Scene | VoiceoverScene):
        # Initialize the section with timing and animation parameters
        self.scene = scene
        self.transform_runtime = 0.5
        self.wait_time = 1

    def _extract_frame_index(self, filename: str) -> int:
        m = re.search(r"(\d+)", filename)
        if not m:
            raise ValueError(f"No digits in {filename!r}")
        return int(m.group(1))
    
    def _prepare_simulations_frames(self):
        # load & sort object renders
        imgs = [f for f in os.listdir("scripts/ReuseExample/images/") if f.endswith(".png")]
        imgs.sort(key=self._extract_frame_index)
        self.frame_list = [
            ImageMobject(f"scripts/ReuseExample/images/{f}") for f in imgs
        ]

        # load & sort matrix renders
        mats = [f for f in os.listdir("scripts/ReuseExample/results/") if f.endswith(".png")]
        mats.sort(key=self._extract_frame_index)
        self.matrix_list = [
            ImageMobject(f"scripts/ReuseExample/results/{f}") for f in mats
        ]
        min_size = min(len(self.frame_list), len(self.matrix_list))
        print(min_size)
        # min_size = 10
        self.sim_frames = [self._show_simulation_frames(i) for i in range(min_size)]
    

    def _show_simulation_frames(self, iteration: int)->Group:
        #Create a VGroup of the frame and the matrix
        frame = self.frame_list[iteration]
        matrix = self.matrix_list[iteration]
        matrix.scale_to_fit_width(frame.get_width())
        matrix.next_to(frame, LEFT, buff=1)
        #Adding a surronding box around the matrix
        matrix_box = SurroundingRectangle(matrix, buff=0.0, color=BLACK, stroke_width=1)
        hessian_label = BraceLabel(matrix_box, text="Hessians", label_constructor=Tex, buff=0.1, font_size=FONT_SIZE).set_color(BLACK)
        rode_twist_label = BraceLabel(frame, text="IPC:RodeTwist", label_constructor=Tex, buff=0.1, font_size=FONT_SIZE).set_color(BLACK)
        return Group(frame, matrix, matrix_box, hessian_label, rode_twist_label)

    def _show_table(self)->VGroup:
        t3 = Table(
            [["220(s)", "76(s)"],
             ["94(s)", "77(s)"]],
            row_labels=[Text("Accelerate", color=BLACK), Text("Parth + Accelerate", color=BLACK)],
            col_labels=[Text("Symbolic", color=BLACK), Text("Numeric", color=BLACK)],
            include_outer_lines=True,
            line_config={"stroke_width": 1, "color": BLACK},
        )
        t3.get_entries().set_color(BLACK)
        t3.get_horizontal_lines().set_color(BLACK)
        t3.get_vertical_lines().set_color(BLACK)
        t3.remove(*t3.get_vertical_lines())
        return t3

    def play_results(self):
        # Entry point for Section 1 animation, with or without voiceover
        if isinstance(self.scene, VoiceoverScene):
            self._prepare_simulations_frames()
            script1 = "By integrating Parth into the Cholesky Solver, now we can reuse the fill-reducing ordering by only updating the permutation vector related to the red region\
                , where the sparsity pattern changes locally and gradually."
            with self.scene.voiceover(text=script1) as tracker:
                frame_and_matrix = self.sim_frames[0]
                frame_and_matrix.scale_to_fit_width(self.scene.camera.frame_width * 0.95)
                frame_and_matrix.center()
                self.scene.add(frame_and_matrix)
                total_time = tracker.duration
                time_per_iteration = total_time / (len(self.frame_list) - 1)
                rt = max(0.2, time_per_iteration)
                for i in range(1, len(self.sim_frames)):
                    new_frame_and_matrix = self.sim_frames[i]
                    new_frame_and_matrix.scale_to_fit_width(self.scene.camera.frame_width * 0.95)
                    new_frame_and_matrix.center()
                    self.scene.remove(frame_and_matrix)
                    self.scene.add(new_frame_and_matrix)
                    self.scene.wait(0.1)
                    frame_and_matrix = new_frame_and_matrix

            self.scene.play(FadeOut(frame_and_matrix), run_time=self.transform_runtime)
            self.scene.wait(self.wait_time)
            table = self._show_table()
            table.scale_to_fit_width(self.scene.camera.frame_width * 0.95)
            table.center()
            script2 = "In this example, by integrating Parth into Apple Accelerate solver, Symbolic runtime is reduced from 220s to 94s\
                without side effect on numerical phase."
            with self.scene.voiceover(text=script2) as tracker:
                self.scene.play(FadeIn(table), run_time=self.transform_runtime)
                self.scene.wait(self.wait_time)
            
            text = Text("See the paper for more results ...", font_size=FONT_SIZE, font=FONT_TYPE, color=BLACK)

        

            # script3 = "We can also reuse the fill-reducing ordering not only in the presence of changes in the graph edges, but also in the graph number of nodes, such as this patching example."
            # with self.scene.voiceover(text=script3) as tracker:
            #     pass

            # script4 = "For this example, acheving 2.9x speedup."
            # with self.scene.voiceover(text=script4) as tracker:
            #     pass

            # script5 = "However, Parth is not a silver bullet, and it has some limitations."
            # with self.scene.voiceover(text=script5) as tracker:
            #     pass

            # script6 = "First, Parth is only when the changes are local. So for example, for this simulation from Arcsim where the changes happen in all over the place, the reuse would be challenging."
            # with self.scene.voiceover(text=script6) as tracker:
            #     pass

            # script7 = "Second, Parth is can provide performance benfits if the fill-reducing ordering is the major bottleneck. However, if the fill-reducing ordering is not the major bottleneck, the performance benfits would be limited."
            # with self.scene.voiceover(text=script7) as tracker:
            #     pass

            # script8 = "As for the future work, Parth is just the beginning, because it is compatible with other symbolic analysis, we expect many more symbolic analysis adapt Parth and become avaiable for dynamic sparsity patterns."
            # with self.scene.voiceover(text=script8) as tracker:
            #     pass

        else:
            pass