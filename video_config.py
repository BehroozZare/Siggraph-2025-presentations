# FastTrack Video Configuration
# This file contains settings and timing for each section of the video

# Total target duration: 20 seconds
# Section timing breakdown:
SECTION_TIMINGS = {
    "section1_sparse_system": 3.5,      # Ax=b introduction
    "section2_newton_solver": 6.5,      # Newton method with slow Cholesky
    "section3_parth_solution": 4.0,     # Parth HGD graph and partial updates
    "section4_code_speedup": 3.0,       # 3 lines of code + speedup
    "section5_outro": 3.0                # Title and link fade to black
}

# Scripts for each section (for reference)
SECTION_SCRIPTS = {
    "section1": "Repetitive solve of a sparse linear solver is important for many applications such as computing second order optimization methods",
    
    "section2": "However, these Cholesky solvers become slow when the underlying sparsity pattern changes continuously due to overhead of permutation vector computation.",
    
    "section3": "In Parth we accelerate fill-reducing ordering analysis by reusing the computation across calls to this kernel.",
    
    "section4": "With only adding 3 lines of code required for using Parth, practitioners can enjoy up to 10x speedup for their Cholesky solve in these dynamic applications.",
    
    "section5": "For more information, visit the following link"
}

# File paths for images/animations
IMAGE_PATHS = {
    "matrices": "Figures/Matrices/test.jpg",
    "animation_base": "Figures/Animation/test.jpg", 
    "animation_progress": "Figures/Animation/test1.jpg"
}

# Rendering settings
RENDER_SETTINGS = {
    "quality": "medium_quality",  # or "low_quality", "high_quality"
    "fps": 15,
    "resolution": "480p"
} 