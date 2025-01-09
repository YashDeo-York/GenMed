from setuptools import setup, find_packages

setup(
    name='generative_metrics_medical_imaging',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'matplotlib'
    ],
    description='A package to evaluate generative metrics in medical imaging.',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/your-username/generative-metrics-medical-imaging',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
