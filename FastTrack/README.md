# FastTrack-Siggraph-2025

FastTrack video created using Manim!

## Overview

This project generates a 20-second video to showcase FastTrack/Parth sparse linear solver acceleration. The video is organized into three modular sections, each implemented in a separate Python file for easy editing and testing.

## File Structure

```
FastTrack-Siggraph-2025/
├── main_fasttrack_video_no_audio.py      # Main file: combines all sections (no audio)
├── main_fasttrack_video_with_audio.py    # Main file: combines all sections (with audio)
├── section1_problem_definition.py        # Section 1: Problem definition & importance
├── section2_parth_solution.py            # Section 2: Parth solution & integration
├── section3_ending.py                    # Section 3: Outro (title & link)
├── Figures/                              # Images and graphics for the video
│   ├── section1/
│   │   └── sad-face.svg
│   ├── section2/
│   │   ├── adapt.png
│   │   ├── adapt.svg
│   │   ├── brain.svg
│   │   ├── FastTrack.png
│   │   └── FastTrack2.png
│   └── section3/
│       └── qr_code.png
├── media/                                # Generated video outputs
├── requirements.txt                      # Python dependencies
└── README.md                             # This documentation
```

## Video Sections

### Section 1: Problem Definition
- **Script**: "Fast execution of a sequence of Sparse Cholesky solves is important in many graphics and scientific applications!"
- **Animation**: Illustrates the Newton algorithm with matrix updates, highlighting the need for fast linear solves.
- **File**: `section1_problem_definition.py`
- **Class**: `ProblemDefinition`

### Section 2: Parth Solution
- **Script**: "With only adding 3 lines of code required for using Parth, practitioners can enjoy up to 10x speedup."
- **Animation**: Shows 3 lines of code, Parth integration, and a speedup visualization.
- **File**: `section2_parth_solution.py`
- **Class**: `ParthContribution`

### Section 3: Outro
- **Script**: "For more information, visit the following link."
- **Animation**: Paper title and GitHub link, fading to black.
- **File**: `section3_ending.py`
- **Class**: `Outro`

## Usage

### Render Complete Video

```bash
# No audio
manim -pql main_fasttrack_video_no_audio.py FastTrackComplete

# With audio
manim -pql main_fasttrack_video_with_audio.py FastTrackComplete
```

### Render Individual Sections

```bash
# Section 1 only
manim -pql section1_problem_definition.py ProblemDefinition

# Section 2 only
manim -pql section2_parth_solution.py ParthContribution

# Section 3 only
manim -pql section3_ending.py Outro
```

## Customization

- **Edit Animations**: Modify the respective `section*.py` files.
- **Change Images**: Add or update images in the `Figures/` directory and update paths in the code.
- **Adjust Timing**: If you have a timing configuration file (e.g., `video_config.py`), edit section durations there.

## Dependencies

- Python 3.7+
- Manim Community Edition
- Required image files in `Figures/`

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Rendering Videos

### Quick Test Render (Low Quality)
To quickly test your video (fast rendering, low resolution), use the `-pql` flag:

```bash
manim -pql main_fasttrack_video_no_audio.py
```
- `-pql` stands for: Preview, Quick, Low quality
- This is useful for development and debugging.

### High-Quality Render (1080p, 60fps)
To generate a high-quality video (1920x1080, 60 frames per second), use the following command:

```bash
manim -pqh --fps 60 --resolution 1920,1080 main_fasttrack_video_no_audio.py FastTrackComplete
```
- `-pqh` stands for: Preview, Quick, High quality
- `--fps 60` sets the frame rate to 60 frames per second
- `--resolution 1920,1080` sets the output resolution to Full HD (1080p)

You can use the same commands for `main_fasttrack_video_with_audio.py` or any section file as needed.

## Development Workflow

1. **Edit individual sections**: Work on each `section*.py` file separately.
2. **Test sections**: Render each section individually to verify timing and content.
3. **Integrate**: Use `main_fasttrack_video_no_audio.py` or `main_fasttrack_video_with_audio.py` to combine all sections.
4. **Fine-tune**: Adjust timing and content as needed. 