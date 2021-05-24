import setuptools
import os
requirementPath = 'gc-venv-green.txt'
install_requires = []  # Examples: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

    setuptools.setup(
        name="gamechangerml",
        version="0.1.0",
        author="Booz Allen Hamilton",
        author_email="ha_robert@example.com",
        description="Package for GAMECHANGER ML modules",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/dod-advana/gamechanger-ml",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3.6",
            "License :: ",
            "Operating System :: OS Independent",
        ],
        python_requires="==3.6.*",
        install_requires=install_requires
    )
