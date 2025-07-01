from setuptools import setup, find_packages

setup(
    name="aiams-tools",
    version="1.0.0",
    packages=find_packages(),
    description="AI-Assisted Manufacturing System Tools",
    author="AIAMS Team",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "ultralytics",
        "matplotlib",
        "pdf2image",
        "pymupdf",
        "pillow",
        "pandas",
        "torch",
        "tqdm",
        "tensorboard",
        "pyyaml",
        "nanoid",
    ],
)
