import os
import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController

def generate_launch_description():
    package_name = 'test_cases_kalman_filter'
    world_file_name = 'test_world.wbt'
    package_dir = get_package_share_directory(package_name)
    robot_description_path = os.path.join(package_dir, 'resource', 'E-puck.urdf')

    # Ruta del archivo del mundo de Webots
    world_path = os.path.join(package_dir, 'worlds', world_file_name)

    # WebotsLauncher permite lanzar una simulación de Webots
    webots = WebotsLauncher(
        world=world_path,
        # ros2_supervisor = True
    )

    # webots_controller = WebotsController(
    #     robot_name='e-puck',
    #     parameters=[
    #         {'robot_description': robot_description_path},
    #     ]
    # )

    # Nodo del controlador del robot (localization_node)
    robot_controller = Node(
        package=package_name,
        executable='robot_controller',
        output='screen',
    )

    return LaunchDescription([
        webots,
        # webots._supervisor,
        # webots_controller,
        robot_controller,
        # Este evento se lanza cuando el nodo de Webots se cierra.
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        )
    ])