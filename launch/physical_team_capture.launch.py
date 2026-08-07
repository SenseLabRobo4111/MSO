"""Record a physical multirobot run without commanding any robot."""

from pathlib import Path
import re

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


RUN_ID_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{2,79}$')


def _capture_setup(context):
    team_size = int(LaunchConfiguration('team_size').perform(context))
    if team_size not in (2, 3, 5):
        raise ValueError('team_size must be 2, 3, or 5')

    run_id = LaunchConfiguration('run_id').perform(context).strip()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            'run_id must contain 3-80 letters, digits, dots, dashes, or '
            'underscores and start with a letter or digit')
    output_root = Path(
        LaunchConfiguration('output_root').perform(context)
    ).expanduser().resolve()
    run_dir = output_root / run_id

    topics = [
        '/tf',
        '/tf_static',
        '/merged_map',
        '/mso/registration_event',
        '/mso/safety_event',
        '/mso/coverage_event',
    ]
    for robot_id in range(team_size):
        prefix = f'/robot_{robot_id}'
        topics.extend([
            f'{prefix}/lidar/points',
            f'{prefix}/odom',
            f'{prefix}/map',
            f'{prefix}/provisional_predicted_map',
            f'{prefix}/provisional_predicted_map_global',
            f'{prefix}/cmd_vel',
            f'/ground_truth/robot_{robot_id}/pose',
        ])

    bag_recorder = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '--storage', 'sqlite3',
            '--max-bag-size', '4000000000',
            '--output', str(run_dir / 'bag'),
            *topics,
        ],
        output='screen',
    )
    event_logger = Node(
        package='sensemap',
        executable='sensemap_event_logger',
        name='physical_team_event_logger',
        output='screen',
        parameters=[{
            'run_id': run_id,
            'output_dir': str(run_dir / 'events'),
            'use_sim_time': False,
        }],
    )
    return [bag_recorder, event_logger]


def generate_launch_description():
    """Build the generic passive capture launch description."""
    return LaunchDescription([
        DeclareLaunchArgument(
            'team_size',
            description='Physical robot count: 2, 3, or 5'),
        DeclareLaunchArgument(
            'run_id',
            description='Unique run identifier prepared before launch'),
        DeclareLaunchArgument(
            'output_root',
            description='Absolute data-volume directory for run folders'),
        OpaqueFunction(function=_capture_setup),
    ])
