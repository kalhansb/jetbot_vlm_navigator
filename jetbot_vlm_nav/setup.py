from setuptools import setup

package_name = 'jetbot_vlm_nav'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kalhan',
    maintainer_email='kalhansandaru@gmail.com',
    description='VLM-based object detection and navigation for JetBot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vlm_navigator_node = src.vlm_navigator_node:main',
        ],
    },
)
