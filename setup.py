from distutils.core import setup
import sys

PACKAGE_DATA = []

# add specific library by platform
if sys.platform.startswith('darwin'):
    PACKAGE_DATA += [
            'bin/*.dylib',
    ]
elif sys.platform.startswith('win32'):
    PACKAGE_DATA += [
            path.join('bin', '*.dll'),    ]
else:
    PACKAGE_DATA += [
            'bin/*.so',
    ]

setup(
    name='gpufuse',
    description='Python wrapper and helper functions for diSPIM fusion on GPU from Shroff Lab',
    version='0.1',
    author='Talley Lambert',
    author_email='talley.lambert@gmail.com',
    url='https://github.com/tlambert03/gpufuse',
    packages=['gpufuse',],
    license='MIT',
    long_description=open('README.md').read(),
    package_data={
        'gpufuse': PACKAGE_DATA,
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Visualization'
    ],
    python_requires='>=3.6',
    install_requires=[
        'A',
        'B'
    ],
    entry_points={
        'console_scripts': [
            'gpufuse = gpufuse'
        ],
    },
)