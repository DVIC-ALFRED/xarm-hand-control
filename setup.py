import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="robot-hand-control",
    version="0.0.1",
    author="Dimitri VINET",
    author_email="dimitri.vinet@outlook.com",
    description="Robot control with hand movements",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    # project_urls={
    #     "Bug Tracker": "https://github.com/dimitrivinet/xarm_hand_control/issues",
    # },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "opencv_contrib_python==4.5.3.56",
        "joblib==1.0.1",
        "mediapipe==0.8.6.2",
    ],
)
