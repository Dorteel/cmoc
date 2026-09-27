from setuptools import find_packages, setup

setup(
    name='cmoc_perception',
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/cmoc_perception']),
        ('share/cmoc_perception', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Project Maintainer',
    maintainer_email='maintainer@example.com',
    description='Architecture-level VLM observation of ROS camera images.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
        'observe_with_vlm_server = cmoc_perception.observe_with_vlm_server:main',
    ]},
)
