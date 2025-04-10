# iSCAT Fast Axial Localization

A Python package for fast axial localization of nano-objects in wide-field interferometric scattering (iSCAT) microscopy. This package implements an approximate model derived from vectorial diffraction theory, combining Bayesian estimation and analytical fitting for efficient axial position determination.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/NTU-Ling-lab/fast-iscat.git
cd fast-iscat
```

2. Set up the environment\
\
Install C++ compiler:\
In Windows 10/11, install Visual Studio Code at https://code.visualstudio.com/download \

In MacOS, install Mac C++ Compiler via terminal
```bash
xcode-select --install
```

\In Ubuntu/Debian Linux:
```bash
sudo apt update
sudo apt install build-essential
```

\In RHEL/CentOS Linux:
```bash
sudo yum group install "Development Tools"
```

3. Install the required dependencies via Anaconda:
```bash
conda env create -f fast-iscat_env.yaml
conda activate fast-iscat_env
```

## Project Structure

```
fast-iscat/
├── fast_iscat/
│   ├── __init__.py
│   ├── core.py           # Core functions
│   ├── utils.py          # Utility functions
│   ├── bayesian_estimation.py  # Bayesian inference implementation
│   ├── fitting.py        # Model fitting routines
│   ├── forward_models.py # iPSF models
│   └── parameters.py     # Parameter management
├── data/                 # Data directory
│   ├── static.mat        # Static sample data
│   ├── dynamic.mat       # Dynamic sample data (thermal expansion)
│   └── diffusion.mat     # Brownian motion sample data
├── config/               # Configuration files
│   ├── parameters_static.json
│   ├── parameters_dynamic.json
│   ├── parameters_diffusion.json
│   └── parameters_documentation.md # Detailed parameter documentation
├── demo.py               # Execution script demo
├── demo.ipynb            # Jupyter notebook demo
├── requirements.txt      # Project dependencies
├── LICENSE.txt           # License information
└── README.md
```

## Sample Types

The package demonstrates axial localization in three types of samples:
1. **Static sample**: Minimal motion, used to establish baseline precision
2. **Dynamic sample**: Thermal expansion-induced displacement
3. **Diffusion sample**: Free Brownian motion in aqueous environment

## Configuration

The package uses JSON configuration files to specify parameters:
- `parameters_static.json`: Configuration for static sample analysis
- `parameters_dynamic.json`: Configuration for dynamic sample analysis
- `parameters_diffusion.json`: Configuration for Brownian motion analysis

Reference documentation is also provided:
- `parameters_documentation.md`: Complete documentation of all parameters with units

Key parameters include:
- Wavelength and numerical aperture
- Particle properties
- Camera settings

## Instructions for Use

Run demo.py in Python IDE (Spyder, PyCharm, etc.) or demo.ipynb in Jupyter Notebook

## Code Formatting

This codebase has been reformatted according to PEP 8 style guidelines for better readability and maintainability using Cursor + Claude-3.7-Sonnet. The improvements include:

- Enhanced docstrings with detailed parameter descriptions
- Consistent code spacing and indentation
- Proper line length limits
- Improved organization of imports and functions
- Cleaner variable naming and code structure

These formatting changes maintain the original functionality while making the code more accessible and easier to understand.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

**Liaoliao Wei**
- Email: [liaoliao001@e.ntu.edu.sg](mailto:liaoliao001@e.ntu.edu.sg)

**Tong Ling**
- Email: [tong.ling@ntu.edu.sg](mailto:tong.ling@ntu.edu.sg)