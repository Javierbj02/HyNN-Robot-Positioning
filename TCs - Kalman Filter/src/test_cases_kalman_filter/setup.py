from setuptools import find_packages, setup

package_name = 'test_cases_kalman_filter'
data_files = []
data_files.append(('share/ament_index/resource_index/packages', ['resource/' + package_name]))
data_files.append(('share/' + package_name + '/launch', ['launch/webots_launch_tc1.py']))
data_files.append(('share/' + package_name + '/launch', ['launch/webots_launch_tc2.py']))
data_files.append(('share/' + package_name + '/launch', ['launch/webots_launch_tc3.py']))
data_files.append(('share/' + package_name + '/worlds', ['worlds/test_world.wbt']))
data_files.append(('share/' + package_name + '/worlds', ['worlds/test_world_2.wbt']))
data_files.append(('share/' + package_name + '/worlds', ['worlds/test_world_3.wbt']))
data_files.append(('share/' + package_name + '/resource', ['resource/E-puck.urdf']))
data_files.append(('share/' + package_name, ['package.xml']))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='javi2002bj',
    maintainer_email='javier.ballesteros@uclm.es',
    description='Package of the Test Cases with the localization algorithms of the Project',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_controller = test_cases_kalman_filter.robot_controller:main'
        ],
    },
)
