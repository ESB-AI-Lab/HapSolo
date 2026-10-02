__author__ = 'mahajrod'
import os
from setuptools import setup, find_packages


dependencies = ["pandas>=1.0", "tqdm>=4.0"]

setup(name='hapsolo',
      version='1.0',
      packages=[],
      author='ESB-AI-Lab',
      install_requires=dependencies,
      scripts=["hapsolo.py", "hapsolo_cli.py", "preprocessfasta.py", "search_orthologs.py"],
      long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),)