import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'sensemap'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='Fei Qiao',
    maintainer_email='qiaofei@tsinghua.edu.cn',
    description='Lightweight local occupancy-map prediction for multirobot exploration',
    license='Proprietary',
    entry_points={
        'console_scripts': [
            'sensemap_predictor = sensemap.predict_map:main'
        ],
    },
)
