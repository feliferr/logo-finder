'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

with open("version") as version_file:
    version = version_file.read().strip()


setup(name='logo_classification_model',
      version=version,
      packages=find_packages(),
      include_package_data=True,
      description='Logo classification model on Cloud ML Engine',
      author='Felipe Ferreira',
      author_email='felipe.ferreira@corp.globo.com',
      license='Unlicense',
      install_requires=[
          'tensorflow-gpu==2.12.0',
          'setuptools==46.4.0',
          'pillow==10.3.0',
          'scikit-learn==1.5.0',
          'keras==2.3.1'],
      zip_safe=False)