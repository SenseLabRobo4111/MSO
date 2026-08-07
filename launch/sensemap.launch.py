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

    declare_crop_size_cmd = DeclareLaunchArgument(
        'crop_size',
        default_value='256',
        description='Positive occupancy crop side length in cells'
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
            'crop_size': LaunchConfiguration('crop_size'),
        }],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    return LaunchDescription([
        declare_robot_id_cmd,
        declare_model_path_cmd,
        declare_architecture_cmd,
        declare_crop_size_cmd,
        start_robot_client,
    ])
