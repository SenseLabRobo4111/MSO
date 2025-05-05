import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, OpaqueFunction, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource, FrontendLaunchDescriptionSource
from launch_ros.actions import Node, PushRosNamespace
from launch.substitutions import LaunchConfiguration
from launch.conditions import LaunchConfigurationEquals

def generate_launch_description():

    declare_robot_id_cmd = DeclareLaunchArgument(
        'robot_id',
        default_value='0',
        description='The id of the robot'
    )

    start_robot_client = Node(
        package='sensemap',
        executable='sensemap_predictor',
        name='sensemap_predictor',
        output='screen',
        parameters=[{'robot_id': LaunchConfiguration('robot_id')}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    ld = LaunchDescription()

    # Add the actions
    ld.add_action(declare_robot_id_cmd)
    ld.add_action(start_robot_client)
    return ld