from setuptools import setup, find_packages


def get_top_level_modules():
    # Collect all top-level .py files in src/
    import glob, os
    top_level_modules = [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob("src/*.py")
    ]
    return top_level_modules

setup(
    name="ReddiCompass", 
    version="0.1.0",
    description="A package for tailored text processing, feature extraction and text classification from Social Media texts",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="honestus",
    author_email="onesto.giuseppe@gmail.com",
    url="https://github.com/honestus/ReddiCompass",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    py_modules=get_top_level_modules(),
    include_package_data=True,
    install_requires=[
        "pandas>=2.3.1",
        "numpy>=1.26.4",
        "scikit-learn==1.6.1",
        "scipy<1.13",
        "eng-spacysentiment",
        "emoji==2.14.1",
        "unidecode",
        "vaderSentiment==3.3.2",
        "emosent-py==0.1.7",
        "nltk==3.9.1",
        "detoxify",
        "textblob==0.19.0",
        "validators==0.35.0",
        "tldextract",
        "pyarrow",
        "gensim",
        "spacy",
        "nrclex @ git+https://github.com/stormbeforesunsetbee/NRCLex.git",
        "Pattern @ git+https://github.com/NicolasBizzozzero/pattern@66ab34453a3443c06a4ebda092c8d1947c83b17a",
        "Moral_Foundation_FrameAxis @ git+https://github.com/edoardo-lucini/Moral_Foundation_FrameAxis.git"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7,<=3.11"
)
