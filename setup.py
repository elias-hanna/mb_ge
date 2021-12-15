# Created by Elias Hanna 
# Date: 22/11/21

from setuptools import setup, find_packages

setup(name='mb_ge',
      install_requires=['gym', 'numpy'],
      version='1.0.0',
      packages=find_packages(),
      #include_package_data=True,
      author="Elias Hanna",
      author_email="h.elias@hotmail.fr",
      description="Model-Based Go-Explore, to solve sparse reward/interaction problems in a data-efficient way",
)
