from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    declare_robot_id_cmd = DeclareLaunchArgument(
        'robot_id',
        default_value='0',
        description='Numeric robot identifier'
    )

    declare_model_path_cmd = DeclareLaunchArgument(
        'model_path',
        description='Absolute path to the distilled MSO checkpoint'
    )

    declare_architecture_cmd = DeclareLaunchArgument(
        'architecture',
        default_value='deconv',
        description='Distilled decoder variant: deconv (paper model) or bilinear'
    )

    start_robot_client = Node(
        package='sensemap',
        executable='sensemap_predictor',
        name='sensemap_predictor',
        output='screen',
        parameters=[{
            'robot_id': LaunchConfiguration('robot_id'),
            'model_path': LaunchConfiguration('model_path'),
            'architecture': LaunchConfiguration('architecture'),
        }],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    return LaunchDescription([
        declare_robot_id_cmd,
        declare_model_path_cmd,
        declare_architecture_cmd,
        start_robot_client,
    ])
