from setuptools import setup, find_packages

setup(
    name='coral',
    version='0.0',
    packages=find_packages(include=["ssm_rl", "envs"]),
    install_requires=["cw2",
                      "dm-control",
                      "gym",
                      "imageio",
                      "matplotlib",
                      "numpy==1.23.0",  # version important for sk-video which is a bit outdated
                      "opencv-python",
                      "sk-video",
                      "scipy",
                      "mani-skill2",
                      "pillow",
                      "PyYAML"],
    url='',
    license='MIT',
    author='anonymous',
    author_email='',
    description=''
)