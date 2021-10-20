import setuptools
from pathlib import Path
from typing import List
import re
import os


ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
REQUIREMENTS_PATH = Path(ROOT_PATH, "k8s.requirements.txt")
DEV_REQUIREMENTS_PATH = Path(ROOT_PATH, "k8s.requirements.txt")
README_PATH = Path(ROOT_PATH, "README.md")


def parse_requirements(requirements: Path) -> List[str]:
    with requirements.open(mode="r") as fd:

        rlist_sans_comments = [
            line.strip()
            for line in fd.read().split("\n")
            if (line.strip() or line.strip().startswith("#"))
        ]

        final_rlist = [
            line
            if not re.match(pattern=r"^https?://.*$", string=line)
            else re.sub(
                pattern=r"(.*(?:https?://.*/)([a-zA-Z0-9_].*)[-]([a-zA-Z0-9.]*)([.]tar[.]gz|[.]tgz).*)",
                repl=r"\2 @ \1",
                string=line,
            )
            for line in rlist_sans_comments
        ]

    return final_rlist


def parse_readme(readme: Path) -> str:
    with readme.open("r") as fh:
        long_description = fh.read()
    return long_description


setuptools.setup(
    name="gamechangerml",
    version="1.0.0",
    author="Booz Allen Hamilton",
    author_email="gamechanger@advana",
    description="Package for GAMECHANGER ML modules",
    long_description=parse_readme(README_PATH),
    long_description_content_type="text/markdown",
    url="https://github.com/dod-advana/gamechanger-ml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6 - 3.8",
        "License :: ",
        "Operating System :: OS Independent",
    ],
    python_requires="~=3.6.*,<3.9.*",
    install_requires=parse_requirements(REQUIREMENTS_PATH),
    include_package_data=True,
    extras_require={"dev": parse_requirements(DEV_REQUIREMENTS_PATH)},
)
