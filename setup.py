"""Installation metadata; all dependency files are shipped in the source archive."""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def requirements(filename):
    """Read one flat dependency list relative to this file."""
    return [
        line.split("#", 1)[0].strip()
        for line in (ROOT / filename).read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    ]


experiment_dependencies = requirements("requirements-experiments.txt")

setup(
    name="TR-Cauchy",
    version="0.1.0",
    description="FAMO-based TR-Cauchy core and multi-task training integrations",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="MIT",
    license_files=("LICENSE", "THIRD_PARTY_NOTICES.md")
    + tuple(str(path.relative_to(ROOT)) for path in sorted((ROOT / "LICENSES").glob("*.txt"))),
    packages=find_packages(include=["methods", "methods.*", "experiments", "experiments.*", "examples", "examples.*"]),
    py_modules=["famo"],
    include_package_data=True,
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=requirements("requirements.txt"),
    extras_require={
        "experiments": experiment_dependencies,
        "toy": experiment_dependencies + ["matplotlib>=3.3", "seaborn>=0.11"],
        "celeba": experiment_dependencies + ["torchvision>=0.15", "Pillow>=8"],
        "qm9": experiment_dependencies + ["torch-geometric>=2.3"],
        "baselines": ["cvxpy>=1.1"],
        "profiling": ["psutil>=5.8"],
        "dev": requirements("requirements-dev.txt"),
    },
)
