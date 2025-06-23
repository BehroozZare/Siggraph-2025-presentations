# FastTrack-Siggraph-2025
Fast track video created using manim!

# FastTrack Video - Organized Structure

This project creates a 20-second video showcasing FastTrack/Parth sparse linear solver acceleration, organized into 3 modular sections that can be designed and modified independently.

## File Structure

```
FastTrack/
├── main_fasttrack_video.py          # Main file combining all sections
├── section1_problem_definition.py   # Section 1: Problem definition & importance
├── section2_parth_solution.py       # Section 2: Parth solution & integration
├── section3_finish.py               # Section 3: Title & link outro
├── video_config.py                  # Configuration & timing settings
├── step_through_algorithm.py        # Original algorithm animation
├── Figures/                         # Images and graphics
│   ├── section2/
│   │   ├── brain.svg
│   │   └── FastTrack.png
├── media/                           # Generated video outputs
└── README.md                        # This documentation
```

## Video Sections (20 seconds total)

### Section 1: Problem Definition (3.5s)
- **Script**: "Fast execution of a sequence of Sparse Cholesky solves is important in many graphics and scientific applications!"
- **Animation**: Shows Newton algorithm with matrix updates, highlighting the importance of fast linear solves
- **File**: `section1_problem_definition.py`
- **Class**: `ProblemDefinition`

### Section 2: Parth Solution (6.5s) 
- **Script**: "With only adding 3 lines of code required for using Parth, practitioners can enjoy up to 10x speedup"
- **Animation**: Shows 3 lines of code, Parth integration with libraries, and speedup visualization
- **File**: `section2_parth_solution.py`
- **Class**: `ParthContribution`

### Section 3: Outro (3.0s)
- **Script**: "For more information, visit the following link"
- **Animation**: Paper title and GitHub link fading to black
- **File**: `section3_finish.py`
- **Class**: `Outro`

## Usage

### Render Complete Video
```bash
manim -pql main_fasttrack_video.py FastTrackComplete
```

### Render Individual Sections
```bash
# Section 1 only
manim -pql section1_problem_definition.py ProblemDefinition

# Section 2 only  
manim -pql section2_parth_solution.py ParthContribution

# Section 3 only
manim -pql section3_finish.py Outro
```

### Test Original Algorithm
```bash
manim -pql step_through_algorithm.py AlgorithmWithTitleAndImages
```

## Customization

### Modify Timing
Edit `video_config.py` to adjust section durations:
```python
SECTION_TIMINGS = {
    "section1_sparse_system": 3.5,    # Adjust as needed
    "section2_newton_solver": 6.5,    # Longest section
    "section3_parth_solution": 4.0,   # Parth solution
    "section4_code_speedup": 3.0,     # Code integration
    "section5_outro": 3.0             # Outro
}
```

### Update Content
Each section file can be modified independently:
- Change animations by editing the respective `section*.py` files
- Update scripts in `video_config.py` 
- Modify image paths in `IMAGE_PATHS` dictionary

### Change Images/Animations
Update the paths in `video_config.py`:
```python
IMAGE_PATHS = {
    "matrices": "path/to/your/matrix/image.jpg",
    "animation_base": "path/to/base/animation.jpg",
    "animation_progress": "path/to/progress/animation.jpg"  
}
```

## Dependencies
- Manim Community Edition
- Python 3.7+
- Required image files in `Figures/` directory

## Development Workflow

1. **Design individual sections**: Work on each `section*.py` file separately
2. **Test sections**: Render individual sections to verify timing and content
3. **Integrate**: Use `main_fasttrack_video.py` to combine all sections
4. **Fine-tune**: Adjust timing in `video_config.py` as needed

This modular approach allows you to:
- Focus on one section at a time
- Easily modify content without affecting other sections  
- Maintain consistent timing across the complete video
- Reuse individual sections in other presentations 