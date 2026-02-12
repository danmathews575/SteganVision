from setuptools import setup, find_packages

setup(
    name="gan-steganography",
    version="0.1.0",
    description="GAN-based Image Steganography with PyTorch",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pillow>=9.5.0",
    ],
)
