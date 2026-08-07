"""Passive JSON event logger for physical-team evidence collection.

The node has subscriptions only. It never publishes motion commands, calls
services, or opens action clients. The experiment stack and independent arena
evaluator publish JSON objects on the configured ``std_msgs/String`` topics.
"""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterable, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


REGISTRATION_STATES = {
    'candidate', 'accepted', 'rejected', 'committed', 'recovered'
}
SAFETY_EVENT_TYPES = {
    'emergency_stop', 'collision', 'near_collision', 'operator_intervention',
    'localisation_loss', 'communication_loss', 'planner_abort',
    'registration_reject', 'registration_rollback', 'other'
}


def _has_keys(value: Dict[str, Any], keys: Iterable[str]) -> bool:
    return all(key in value for key in keys)


def validate_registration(value: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate the fields needed to audit one registration transition."""
    required = (
        'schema_version', 'event_id', 'run_id', 'timestamp_ns',
        'source_robot_id', 'target_robot_id', 'state', 'estimate', 'gate'
    )
    if not _has_keys(value, required):
        return False, 'missing required registration field'
    if value['schema_version'] != '1.0':
        return False, 'unsupported schema version'
    if not isinstance(value['timestamp_ns'], int) or value['timestamp_ns'] < 1:
        return False, 'timestamp_ns must be a positive integer'
    if not isinstance(value['source_robot_id'], int) or not isinstance(
            value['target_robot_id'], int):
        return False, 'robot identifiers must be integers'
    if value['state'] not in REGISTRATION_STATES:
        return False, 'unsupported registration state'
    if value['source_robot_id'] == value['target_robot_id']:
        return False, 'source and target robots must differ'
    estimate = value.get('estimate')
    if not isinstance(estimate, dict) or not _has_keys(
            estimate, ('tx_m', 'ty_m', 'yaw_deg')):
        return False, 'estimate must contain tx_m, ty_m, and yaw_deg'
    if not all(isinstance(estimate[key], (int, float))
               for key in ('tx_m', 'ty_m', 'yaw_deg')):
        return False, 'estimate values must be numeric'
    if not isinstance(value.get('gate'), dict):
        return False, 'gate must be an object'
    return True, ''


def validate_safety(value: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate one observed safety event."""
    required = (
        'schema_version', 'event_id', 'run_id', 'timestamp_ns', 'robot_id',
        'event_type', 'action', 'detail'
    )
    if not _has_keys(value, required):
        return False, 'missing required safety field'
    if value['schema_version'] != '1.0':
        return False, 'unsupported schema version'
    if not isinstance(value['timestamp_ns'], int) or value['timestamp_ns'] < 1:
        return False, 'timestamp_ns must be a positive integer'
    if value['event_type'] not in SAFETY_EVENT_TYPES:
        return False, 'unsupported safety event type'
    return True, ''


def validate_coverage(value: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate one independent coverage sample."""
    required = ('schema_version', 'run_id', 'timestamp_ns', 'coverage')
    if not _has_keys(value, required):
        return False, 'missing required coverage field'
    if value['schema_version'] != '1.0':
        return False, 'unsupported schema version'
    if not isinstance(value['timestamp_ns'], int) or value['timestamp_ns'] < 1:
        return False, 'timestamp_ns must be a positive integer'
    try:
        coverage = float(value['coverage'])
    except (TypeError, ValueError):
        return False, 'coverage must be numeric'
    if not 0.0 <= coverage <= 1.0:
        return False, 'coverage must be within [0, 1]'
    return True, ''


class PassiveEventLogger(Node):
    """Write validated evidence events to append-only JSONL files."""

    def __init__(self) -> None:
        super().__init__('physical_team_event_logger')
        self.declare_parameter('run_id', '')
        self.declare_parameter('output_dir', '')
        self.declare_parameter(
            'registration_topic', '/mso/registration_event')
        self.declare_parameter('safety_topic', '/mso/safety_event')
        self.declare_parameter('coverage_topic', '/mso/coverage_event')

        self.run_id = str(self.get_parameter('run_id').value).strip()
        output_dir = str(self.get_parameter('output_dir').value).strip()
        if not self.run_id:
            raise ValueError('run_id is required')
        if not output_dir:
            raise ValueError('output_dir is required')

        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._files = {
            'registration': open(
                self.output_dir / 'registration_events.jsonl',
                'a', encoding='utf-8', buffering=1),
            'safety': open(
                self.output_dir / 'safety_events.jsonl',
                'a', encoding='utf-8', buffering=1),
            'coverage': open(
                self.output_dir / 'coverage_events.jsonl',
                'a', encoding='utf-8', buffering=1),
            'invalid': open(
                self.output_dir / 'invalid_events.jsonl',
                'a', encoding='utf-8', buffering=1),
        }
        qos = QoSProfile(
            depth=1000,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        topic_specs = (
            ('registration', 'registration_topic', validate_registration),
            ('safety', 'safety_topic', validate_safety),
            ('coverage', 'coverage_topic', validate_coverage),
        )
        self._subscriptions = []
        for kind, parameter, validator in topic_specs:
            topic = str(self.get_parameter(parameter).value)
            subscription = self.create_subscription(
                String,
                topic,
                lambda msg, k=kind, t=topic, v=validator:
                    self._on_event(k, t, v, msg),
                qos,
            )
            self._subscriptions.append(subscription)
        self.get_logger().info(
            f'Passive event logging active for run {self.run_id}')

    def _write(self, kind: str, value: Dict[str, Any]) -> None:
        handle = self._files[kind]
        handle.write(json.dumps(value, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())

    def _on_event(self, kind, topic, validator, message: String) -> None:
        receipt_time_ns = time.time_ns()
        try:
            value = json.loads(message.data)
        except json.JSONDecodeError as error:
            self._write('invalid', {
                'receipt_time_ns': receipt_time_ns,
                'topic': topic,
                'reason': f'invalid JSON: {error}',
                'raw': message.data,
            })
            return
        if not isinstance(value, dict):
            self._write('invalid', {
                'receipt_time_ns': receipt_time_ns,
                'topic': topic,
                'reason': 'event payload is not an object',
                'raw': value,
            })
            return
        valid, reason = validator(value)
        if value.get('run_id') != self.run_id:
            valid = False
            reason = 'run_id does not match logger configuration'
        value['receipt_time_ns'] = receipt_time_ns
        value['recorded_topic'] = topic
        if valid:
            self._write(kind, value)
        else:
            value['validation_error'] = reason
            self._write('invalid', value)

    def destroy_node(self):
        """Flush and close evidence files before node shutdown."""
        for handle in self._files.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        return super().destroy_node()


def main(args=None) -> None:
    """Run the passive ROS 2 event logger."""
    rclpy.init(args=args)
    node = PassiveEventLogger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
