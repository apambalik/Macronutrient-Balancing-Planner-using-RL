"""
Setup script for RL Meal Planning System.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rl-meal-planning",
    version="1.0.0",
    author="RL Meal Planning Team",
    author_email="team@rlmealplanning.com",
    description="An adaptive reinforcement learning agent for personalized macronutrient meal planning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Macronutrient-Balancing-Planner-using-RL",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rl-meal-planning=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords="reinforcement-learning, meal-planning, nutrition, machine-learning, ai",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/Macronutrient-Balancing-Planner-using-RL/issues",
        "Source": "https://github.com/yourusername/Macronutrient-Balancing-Planner-using-RL",
        "Documentation": "https://github.com/yourusername/Macronutrient-Balancing-Planner-using-RL/docs",
    },
) 