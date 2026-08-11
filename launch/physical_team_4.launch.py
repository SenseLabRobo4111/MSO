"""Four-robot physical capture wrapper."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Include the passive capture launch with a four-robot team."""
    generic = Path(get_package_share_directory('sensemap')) / 'launch' / (
        'physical_team_capture.launch.py')
    return LaunchDescription([
        DeclareLaunchArgument('run_id'),
        DeclareLaunchArgument('output_root'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(generic)),
            launch_arguments={
                'team_size': '4',
                'run_id': LaunchConfiguration('run_id'),
                'output_root': LaunchConfiguration('output_root'),
            }.items(),
        ),
    ])
