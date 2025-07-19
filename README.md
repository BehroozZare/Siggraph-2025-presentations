# FastTrack-Siggraph-2025

A comprehensive repository containing video generation projects for FastTrack/Parth sparse linear solver acceleration, created using Manim for SIGGRAPH 2025.

## 📁 Project Structure

This repository contains two main components:

```
FastTrack-Siggraph-2025/
├── FastTrack/                    # Main FastTrack video project
│   ├── main_fasttrack_video_no_audio.py
│   ├── main_fasttrack_video_with_audio.py
│   ├── section1_problem_definition.py
│   ├── section2_parth_solution.py
│   ├── section3_ending.py
│   ├── Figures/                  # Graphics and images
│   ├── media/                    # Generated video outputs
│   ├── requirements.txt
│   └── README.md
├── Presentation/                  # Presentation video project
│   ├── presentation_video_no_audio.py
│   ├── presentation_video_with_audio.py
│   ├── section0_title.py
│   ├── section1_background.py
│   ├── section2_problem_definition.py
│   ├── section3_parth_solution.py
│   ├── section4_ending.py
│   ├── Figures/                  # Graphics and images
│   ├── media/                    # Generated video outputs
│   ├── requirements.txt
│   └── README.md
├── .gitignore
└── README.md                     # This file
```

## 🎬 Video Projects

### 1. FastTrack Video (`FastTrack/`)
A 20-second promotional video showcasing FastTrack/Parth sparse linear solver acceleration. The video is organized into three modular sections:

- **Section 1**: Problem definition and importance of fast sparse Cholesky solves
- **Section 2**: Parth solution demonstration with 3-line code integration
- **Section 3**: Outro with paper title and links

### 2. Presentation Video (`Presentation/`)
A comprehensive presentation video with extended content, organized into five sections:

- **Section 0**: Title and introduction
- **Section 1**: Background and context
- **Section 2**: Problem definition
- **Section 3**: Parth solution details
- **Section 4**: Conclusion and resources

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Manim Community Edition

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd FastTrack-Siggraph-2025

# Install dependencies for FastTrack project
cd FastTrack
pip install -r requirements.txt

# Install dependencies for Presentation project
cd ../Presentation
pip install -r requirements.txt
```

### Rendering Videos

#### FastTrack Video
```bash
cd FastTrack

# Quick test render (low quality)
manim -pql main_fasttrack_video_no_audio.py FastTrackComplete

# High-quality render (1080p, 60fps)
manim -pqh --fps 60 --resolution 1920,1080 main_fasttrack_video_no_audio.py FastTrackComplete
```

#### Presentation Video
```bash
cd Presentation

# Quick test render (low quality)
manim -pql presentation_video_no_audio.py PresentationComplete

# High-quality render (1080p, 60fps)
manim -pqh --fps 60 --resolution 1920,1080 presentation_video_no_audio.py PresentationComplete
```

## 📋 Features

- **Modular Design**: Each video section is implemented in separate Python files for easy editing and testing
- **Multiple Quality Options**: Support for both quick preview renders and high-quality production renders
- **Audio Support**: Versions with and without audio tracks
- **Customizable Graphics**: Easy-to-update figures and images in dedicated directories
- **Cross-Platform**: Works on Windows, macOS, and Linux

## 🛠️ Development

### Editing Videos
1. **Modify Sections**: Edit the respective `section*.py` files in each project
2. **Update Graphics**: Add or replace images in the `Figures/` directories
3. **Test Changes**: Render individual sections for quick feedback
4. **Combine**: Use the main files to combine all sections

### File Organization
- `Figures/`: Contains all graphics, images, and visual assets
- `media/`: Generated video outputs (gitignored)
- `section*.py`: Individual video sections
- `main_*.py`: Complete video compositions

## 📚 Documentation

Each project directory contains its own detailed README with:
- Specific usage instructions
- Detailed file structure
- Customization guidelines
- Development workflows

- [FastTrack README](FastTrack/README.md)
- [Presentation README](Presentation/README.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test your modifications
5. Submit a pull request

## 📄 License

This project is licensed under the terms specified in the LICENSE files within each project directory.

## 🔗 Related Links

- [Manim Documentation](https://docs.manim.community/)
- [SIGGRAPH 2025](https://s2025.siggraph.org/)

---

**Note**: Generated video files are stored in the `media/` directories and are excluded from version control to keep the repository size manageable. 