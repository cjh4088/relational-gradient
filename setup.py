from setuptools import setup, find_packages

setup(
    name='relational-gradient',
    version='0.7.0',
    author='Pi Xi',
    author_email='xiapi@openclaw.ai',
    description='Relational Gradient: Collective Optimization Beyond Adam',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/xiapi-ai/relational-gradient',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    install_requires=['numpy>=1.20.0', 'torch>=1.9.0'],
)
