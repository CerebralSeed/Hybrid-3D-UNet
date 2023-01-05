from setuptools import setup, find_packages

exec(open('hybrid3dunet/version.py').read())

setup(
  name = 'hybrid3dunet',
  packages = find_packages(),
  version = __version__,
  license='MIT',
  description = 'Hybrid 3D Unet - Pytorch',
  author = 'Jeremiah Johnson',
  author_email = 'j.johnson.bbt@gmail.com',
  url = 'https://github.com/CerebralSeed/Hybrid-3D-UNet',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'generative models'
    '3d unet hybrid'
  ],
  install_requires=[
    'accelerate',
    'einops',
    'ema-pytorch',
    'pillow',
    'torch',
    'torchvision',
    'tqdm'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
