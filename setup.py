import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'sensemap'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('lib', package_name, 'config'), glob('sensemap/config/*')),
        (os.path.join('lib', package_name, 'explore_model'), glob('sensemap/explore_model/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='azusa',
    maintainer_email='2723687563@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensemap_predictor = sensemap.predict_map:main'
        ],
    },
)
