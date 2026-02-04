from setuptools import setup, find_packages

setup(
    name='oragent',
    version='0.1.0',
    author='Qi Liu',
    author_email='liuqi_tj@hotmail.com',
    description='Open Research Agent.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='',
    package_dir={'': 'src'},  # Tell setuptools packages are under src/
    packages=find_packages(where='src'),  # Look for packages in src/
    install_requires=[
        # List your project dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'oragent=oragent.cli:main',
        ],
    },
)