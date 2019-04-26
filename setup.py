import setuptools

# The text of the README file
with open("README.md", "r") as fh:
    long_description = fh.read()

# This call to setup() does all the work
setuptools.setup(
    name="bidaf-keras",
    version="1.0.0",
    description="Implementation of Bidirectional Attention Flow for Machine Comprehension in Keras 2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ParikhKadam/bidaf-keras",
    download_url="https://github.com/ParikhKadam/bidaf-keras/archive/v-1.0.0.tar.gz",
    author="Kadam Parikh",
    author_email="parikhkadam@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=["pymagnitude", "keras", "tqdm", "nltk"],
    extras_require={
        "cpu": ["tensorflow"],
        "gpu": ["tensorflow-gpu"],
    },
    entry_points={
        "console_scripts": [
            "bidaf-keras=bidaf.__main__:main",
        ]
    },
)
