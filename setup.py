import setuptools

def readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

setuptools.setup(
    name="ChessVideoAI",
    version="0.0.1",
    author="Douwe Spaanderman",
    author_email="dspaanderman@gmail.com",
    description="Deep learning project to analyse chess games from image or video",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Douwe-Spaanderman/ChessVideoAI",
    project_urls={
        "Bug Tracker": "https://github.com/Douwe-Spaanderman/ChessVideoAI/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.7.3',
)