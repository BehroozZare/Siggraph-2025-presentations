from manim import *

# Configure LaTeX template to include xcolor package
config.tex_template.add_to_preamble(r"\usepackage{xcolor}")

class AnimatedImage(Group):
    def __init__(self, file_list, *, scale=1.0, time_per_frame=0.1, **kwargs):
        # load & scale all frames
        frames = [ImageMobject(f).scale(scale) for f in file_list]
        super().__init__(*frames, **kwargs)
        for i, frame in enumerate(self.submobjects):
            frame.set_opacity(1 if i == 0 else 0)
        self.time_per_frame = time_per_frame
        self.elapsed = 0.0
        # add the updater
        self.add_updater(self._cycle_frames)

    def _cycle_frames(self, mob, dt):
        # mob is the AnimatedImage instance (same as self)
        self.elapsed += dt
        idx = int(self.elapsed / self.time_per_frame) % len(self.submobjects)
        for i, frame in enumerate(self.submobjects):
            frame.set_opacity(1 if i == idx else 0)
        return mob


class AlgorithmWithTitleAndImages(Scene):
    def make_title(self) -> Tex:
        title = Tex(r"Newton Solver with Sparse Cholesky", font_size=48)
        title.to_edge(UP)
        return title

    def make_algorithm_block(self) -> VGroup:
        template = TexTemplate()
        template.add_to_preamble(r"\usepackage{algorithmic}")
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
            line = rf"{i+1}.\quad" + r"\quad"*indent + " " + txt
            lines.append(Tex(line, tex_template=template, font_size=36))
        return Group(*lines).arrange(DOWN, aligned_edge=LEFT, buff=0.3)

    def make_matrix_gif(self) -> VGroup:
        # assume you have, e.g., img1_0.png, img1_1.png, img1_2.png ... etc.
        return AnimatedImage(
            ["Figures/Matrices/test.jpg", "Figures/Matrices/test.jpg", "Figures/Matrices/test.jpg"],
            scale=0.2,
            time_per_frame=0.2,
        )

    def make_animation_gif(self) -> AnimatedImage:
        return AnimatedImage(
            ["Figures/Animation/test.jpg", "Figures/Animation/test.jpg", "Figures/Animation/test1.jpg"],
            scale=0.2,
            time_per_frame=0.2,
        )

    def construct(self):
        title = self.make_title()
        algo = self.make_algorithm_block()
        center = self.make_matrix_gif()
        extra = self.make_animation_gif()

        # layout
        self.add(title)
        # Position algorithm in left 1/3 of scene
        algo.move_to([-4, -0.5, 0])
        # Position center image in top-right quadrant
        center.move_to([3, 1.5, 0]) 
        # Position extra image in bottom-right quadrant
        extra.move_to([3, -1.5, 0])

        # show
        self.play(FadeIn(algo, lag_ratio=0.1))
        self.play(FadeIn(center, shift=UP))
        self.play(FadeIn(extra, shift=UP))

        # pointer animation
        pointer = Tex(r"$\to$", font_size=48)
        pointer.next_to(algo[0], LEFT, buff=0.1)
        self.add(pointer)
        schedule = [(1, 1), (2, 1), (3, 1)]
        for idx, pause in schedule:
            self.play(
                pointer.animate.next_to(algo[idx], LEFT, buff=0.1),
                run_time=0.5
            )
            self.wait(pause)

        self.wait()
