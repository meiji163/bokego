from setuptools import setup
import os
import io

here = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
setup(
    name = "bokego",
    version = "0.3",
    description = "A 9x9 Go engine",
    long_description = long_description,
    author = "meiji163, kyleschan, dukehhong", 
    author_email="mysatellite99@gmail.com",
    packages=["bokego"],
    install_requires=["torch","numpy", "pandas"],
    data_files=[("bokego", ["data/weights/value_1.pt", 
                "data/weights/policy_0.pt",
                 "data/weights/policy_17.pt"]) ],
    python_requires=">=3.7.0",
    license="MIT",
    url="https://github.com/meiji163/BokeGo",
    classifiers = [
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Environment :: GPU :: NVIDIA CUDA',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
        ]
)
