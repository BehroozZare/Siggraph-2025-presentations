# FastTrack Video - Organized Structure

This project creates a 20-second video showcasing FastTrack/Parth sparse linear solver acceleration, organized into 5 modular sections that can be designed and modified independently.

## File Structure

```
FastTrack/
├── main_fasttrack_video.py          # Main file combining all sections
├── section1_sparse_system.py        # Section 1: Ax=b introduction  
├── section2_newton_solver.py        # Section 2: Newton solver problems
├── section3_parth_solution.py       # Section 3: Parth HGD solution
├── section4_simple.py               # Section 4: Code integration & speedup
├── section5_outro.py                # Section 5: Title & link outro
├── video_config.py                  # Configuration & timing settings
├── step_through_algorithm.py        # Original algorithm animation
└── README_FastTrack_Video.md        # This documentation
```

## Video Sections (20 seconds total)

### Section 1: Sparse Linear Systems (3.5s)
- **Script**: "Repetitive solve of a sparse linear solver is important for many applications such as computing second order optimization methods"
- **Animation**: Shows Ax=b equation with sparse matrix visualization
- **File**: `section1_sparse_system.py`

### Section 2: Newton Solver Problems (6.5s) 
- **Script**: "However, these Cholesky solvers become slow when the underlying sparsity pattern changes continuously due to overhead of permutation vector computation"
- **Animation**: Newton algorithm with matrix updates, slow on linear solve step
- **File**: `section2_newton_solver.py`
- **Based on**: Original `AlgorithmWithTitleAndImages` class

### Section 3: Parth Solution (4.0s)
- **Script**: "In Parth we accelerate fill-reducing ordering analysis by reusing the computation across calls to this kernel"
- **Animation**: HGD graph with partially updated ordering vector
- **File**: `section3_parth_solution.py`

### Section 4: Code Integration (3.0s)
- **Script**: "With only adding 3 lines of code required for using Parth, practitioners can enjoy up to 10x speedup"  
- **Animation**: Shows 3 lines of code and speedup visualization
- **File**: `section4_simple.py`

### Section 5: Outro (3.0s)
- **Script**: "For more information, visit the following link"
- **Animation**: Title and GitHub link fading to black
- **File**: `section5_outro.py`

## Usage

### Render Complete Video
```bash
manim -pql main_fasttrack_video.py FastTrackComplete
```

### Render Individual Sections
```bash
# Section 1 only
manim -pql section1_sparse_system.py SparseLinearSystem

# Section 2 only  
manim -pql section2_newton_solver.py NewtonSolverWithUpdates

# Section 3 only
manim -pql section3_parth_solution.py ParthSolution

# Section 4 only
manim -pql section4_simple.py CodeAndSpeedup

# Section 5 only
manim -pql section5_outro.py Outro
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
    # ... etc
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