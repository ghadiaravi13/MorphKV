from setuptools import setup, find_packages

long_description = "MorphKV: A library for efficient key-value cache management in transformer models"

setup(
    name="morphkv",
    version="0.1.0",
    author="Ravi Ghadia",
    author_email="rghadia@utexas.edu",
    description="A library for efficient key-value cache management in transformer models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ghadiaravi13/MorphKV",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers==4.45.0",
        "numpy",
        "plotly",
    ],
    extras_require={
        "flash": [
            "flash-attn>=2.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)