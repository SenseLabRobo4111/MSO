#!/usr/bin/env python3
"""Publish and receive a sequenced best-effort ROS 2 network probe."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time


IDENTIFIER = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$')


class EventWriter:
    """Append probe events to durable JSONL."""

    def __init__(self, path, common):
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.handle = output.open('a', encoding='utf-8', buffering=1)
        self.common = common

    def write(self, event, **fields):
        row = {
            **self.common,
            'event': event,
            'timestamp_ns': time.time_ns(),
            'created_utc': datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        self.handle.write(json.dumps(row, sort_keys=True) + '\n')
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self):
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--session-id')
    parser.add_argument('--robot-id', type=int, required=True)
    parser.add_argument('--peer-robot-id', type=int, required=True)
    parser.add_argument('--publish-topic', required=True)
    parser.add_argument('--subscribe-topic', required=True)
    parser.add_argument('--log-path', required=True)
    parser.add_argument('--start-at-ns', type=int, required=True)
    parser.add_argument('--duration-s', type=float, required=True)
    parser.add_argument('--rate-hz', type=float, default=20.0)
    parser.add_argument('--payload-bytes', type=int, default=256)
    return parser.parse_args()


def make_payload(args, sequence, timestamp_ns):
    """Build a JSON payload at least as large as the declared target."""
    value = {
        'schema_version': '1.0',
        'run_id': args.run_id,
        'source_robot_id': args.robot_id,
        'target_robot_id': args.peer_robot_id,
        'sequence': sequence,
        'sender_timestamp_ns': timestamp_ns,
        'filler': '',
    }
    encoded = json.dumps(value, separators=(',', ':'), sort_keys=True)
    missing = max(0, args.payload_bytes - len(encoded.encode('utf-8')))
    value['filler'] = 'x' * missing
    return json.dumps(value, separators=(',', ':'), sort_keys=True)


def main():
    args = parse_args()
    if not IDENTIFIER.fullmatch(args.run_id):
        raise SystemExit('invalid run-id')
    session_id = args.session_id or f'{args.run_id}.probe.r{args.robot_id}'
    if not IDENTIFIER.fullmatch(session_id):
        raise SystemExit('invalid session-id')
    if args.robot_id < 0 or args.peer_robot_id < 0 or (
            args.robot_id == args.peer_robot_id):
        raise SystemExit('robot identifiers must be distinct')
    if args.publish_topic == args.subscribe_topic or not all(
            topic.startswith('/')
            for topic in (args.publish_topic, args.subscribe_topic)):
        raise SystemExit('probe topics must be distinct absolute names')
    if args.duration_s <= 0 or args.rate_hz <= 0 or args.payload_bytes < 64:
        raise SystemExit('invalid duration, rate, or payload size')
    if args.start_at_ns < time.time_ns() + 5_000_000_000:
        raise SystemExit('start-at-ns must be at least five seconds ahead')

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String
    except ImportError as error:
        raise SystemExit('ROS 2 rclpy and std_msgs are required') from error

    common = {
        'schema_version': '1.0',
        'run_id': args.run_id,
        'session_id': session_id,
        'clock_basis': 'system_utc',
    }
    writer = EventWriter(args.log_path, common)

    class ProbeNode(Node):
        """One directional publisher and reciprocal receiver."""

        def __init__(self):
            super().__init__(f'mso_network_probe_{args.robot_id}')
            qos = QoSProfile(
                depth=1000,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE)
            self.publisher = self.create_publisher(
                String, args.publish_topic, qos)
            self.subscription = self.create_subscription(
                String, args.subscribe_topic, self.receive, qos)
            self.received = 0

        def receive(self, message):
            receipt_ns = time.time_ns()
            try:
                value = json.loads(message.data)
                required = {
                    'run_id', 'source_robot_id', 'target_robot_id',
                    'sequence', 'sender_timestamp_ns'}
                if not isinstance(value, dict) or not required.issubset(value):
                    raise ValueError('missing probe payload fields')
                if value['run_id'] != args.run_id or int(
                        value['source_robot_id']) != args.peer_robot_id or int(
                        value['target_robot_id']) != args.robot_id:
                    raise ValueError('probe identity mismatch')
                self.received += 1
                writer.write(
                    'received', timestamp_ns=receipt_ns,
                    source_robot_id=args.peer_robot_id,
                    target_robot_id=args.robot_id,
                    topic=args.subscribe_topic,
                    sequence=int(value['sequence']),
                    sender_timestamp_ns=int(value['sender_timestamp_ns']),
                    payload_bytes=len(message.data.encode('utf-8')))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                writer.write(
                    'invalid_received', source_robot_id=args.peer_robot_id,
                    target_robot_id=args.robot_id,
                    topic=args.subscribe_topic, detail=str(error))

    rclpy.init()
    node = ProbeNode()
    sent = 0
    end_ns = args.start_at_ns + int(args.duration_s * 1e9)
    interval_ns = int(1e9 / args.rate_hz)
    next_send_ns = args.start_at_ns
    writer.write(
        'session_started', source_robot_id=args.robot_id,
        target_robot_id=args.peer_robot_id, topic=args.publish_topic,
        rate_hz=args.rate_hz, duration_s=args.duration_s)
    try:
        while time.time_ns() < end_ns:
            now_ns = time.time_ns()
            if now_ns >= next_send_ns:
                payload = make_payload(args, sent, now_ns)
                message = String()
                message.data = payload
                node.publisher.publish(message)
                writer.write(
                    'sent', timestamp_ns=now_ns,
                    source_robot_id=args.robot_id,
                    target_robot_id=args.peer_robot_id,
                    topic=args.publish_topic, sequence=sent,
                    sender_timestamp_ns=now_ns,
                    payload_bytes=len(payload.encode('utf-8')))
                sent += 1
                next_send_ns += interval_ns
            rclpy.spin_once(node, timeout_sec=0.005)
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.01)
    finally:
        writer.write(
            'session_completed', source_robot_id=args.robot_id,
            target_robot_id=args.peer_robot_id, topic=args.publish_topic,
            sent_count=sent, received_count=node.received)
        node.destroy_node()
        rclpy.shutdown()
        writer.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
