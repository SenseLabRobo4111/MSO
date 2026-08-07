#!/usr/bin/env python3
"""Aggregate matched N=2/3/5 physical runs without inventing uncertainty."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys


TEAM_SIZES = (2, 3, 5)


def mean(values):
    """Return a mean or null for an empty collection."""
    return statistics.mean(values) if values else None


def sample_sd(values):
    """Return sample standard deviation when at least two runs exist."""
    return statistics.stdev(values) if len(values) > 1 else None


def median(values):
    """Return a median or null for an empty collection."""
    return statistics.median(values) if values else None


def numeric(rows, key):
    """Extract non-null numeric values from run rows."""
    return [float(row[key]) for row in rows if row.get(key) is not None]


def write_csv(path, rows, fieldnames):
    """Write a table with a stable column order."""
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--min-trials-per-team', type=int, default=5)
    return parser.parse_args()


def main():
    """Aggregate complete matched campaigns and fail on incomplete evidence."""
    args = parse_args()
    campaign_root = args.campaign_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    rejected_runs = []
    for audit_path in sorted(campaign_root.rglob('analysis/run_audit.json')):
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        if not audit.get('protocol_diagnostic_pass'):
            rejected_runs.append({
                'run_id': audit.get('run_id'),
                'path': str(audit_path),
            })
            continue
        coverage = audit.get('coverage') or {}
        registration = audit.get('registration') or {}
        run_rows.append({
            'run_id': audit['run_id'],
            'method': audit['method'],
            'site_id': audit['site_id'],
            'arena_id': audit['arena_id'],
            'trial_id': int(audit['trial_id']),
            'team_size': int(audit['team_size']),
            'final_coverage': coverage.get('final_coverage'),
            't80_s': coverage.get('t80_s'),
            'normalised_auc': coverage.get('normalised_auc'),
            'translation_error_median_m': registration.get(
                'translation_error_median_m'),
            'yaw_error_median_deg': registration.get('yaw_error_median_deg'),
            'accepted': registration.get('accepted', 0),
            'rejected': registration.get('rejected', 0),
            'false_accept': registration.get('false_accept', 0),
            'false_reject': registration.get('false_reject', 0),
            'wrong_commits': registration.get('wrong_commits', 0),
        })

    groups = defaultdict(list)
    for row in run_rows:
        groups[(row['method'], row['site_id'], row['arena_id'])].append(row)

    group_audits = []
    summary_rows = []
    for key, rows in sorted(groups.items()):
        by_team = {size: [] for size in TEAM_SIZES}
        for row in rows:
            if row['team_size'] in by_team:
                by_team[row['team_size']].append(row)
        trial_sets = {
            size: {row['trial_id'] for row in team_rows}
            for size, team_rows in by_team.items()
        }
        matched_trials = set.intersection(*trial_sets.values())
        complete = (
            len(matched_trials) >= args.min_trials_per_team
            and all(len(by_team[size]) >= args.min_trials_per_team
                    for size in TEAM_SIZES))
        group_audits.append({
            'method': key[0],
            'site_id': key[1],
            'arena_id': key[2],
            'team_counts': {
                str(size): len(by_team[size]) for size in TEAM_SIZES},
            'matched_trial_ids': sorted(matched_trials),
            'complete': complete,
        })
        if not complete:
            continue
        for size in TEAM_SIZES:
            matched_rows = [
                row for row in by_team[size]
                if row['trial_id'] in matched_trials]
            coverage = numeric(matched_rows, 'final_coverage')
            t80 = numeric(matched_rows, 't80_s')
            auc = numeric(matched_rows, 'normalised_auc')
            trans = numeric(matched_rows, 'translation_error_median_m')
            yaw = numeric(matched_rows, 'yaw_error_median_deg')
            summary_rows.append({
                'method': key[0],
                'site_id': key[1],
                'arena_id': key[2],
                'team_size': size,
                'n_matched_runs': len(matched_rows),
                'final_coverage_mean': mean(coverage),
                'final_coverage_sd': sample_sd(coverage),
                'final_coverage_median': median(coverage),
                't80_s_mean': mean(t80),
                't80_s_sd': sample_sd(t80),
                'normalised_auc_mean': mean(auc),
                'normalised_auc_sd': sample_sd(auc),
                'translation_error_median_of_runs_m': median(trans),
                'yaw_error_median_of_runs_deg': median(yaw),
                'accepted_total': sum(row['accepted'] for row in matched_rows),
                'rejected_total': sum(row['rejected'] for row in matched_rows),
                'false_accept_total': sum(
                    row['false_accept'] for row in matched_rows),
                'false_reject_total': sum(
                    row['false_reject'] for row in matched_rows),
                'wrong_commits_total': sum(
                    row['wrong_commits'] for row in matched_rows),
            })

    run_fields = (
        'run_id', 'method', 'site_id', 'arena_id', 'trial_id', 'team_size',
        'final_coverage', 't80_s', 'normalised_auc',
        'translation_error_median_m', 'yaw_error_median_deg', 'accepted',
        'rejected', 'false_accept', 'false_reject', 'wrong_commits')
    summary_fields = (
        'method', 'site_id', 'arena_id', 'team_size', 'n_matched_runs',
        'final_coverage_mean', 'final_coverage_sd', 'final_coverage_median',
        't80_s_mean', 't80_s_sd', 'normalised_auc_mean',
        'normalised_auc_sd', 'translation_error_median_of_runs_m',
        'yaw_error_median_of_runs_deg', 'accepted_total', 'rejected_total',
        'false_accept_total', 'false_reject_total', 'wrong_commits_total')
    write_csv(output_dir / 'campaign_runs.csv', run_rows, run_fields)
    write_csv(output_dir / 'campaign_summary.csv', summary_rows, summary_fields)

    campaign_protocol_complete = any(
        group['complete'] for group in group_audits)
    report = {
        'schema_version': '1.0',
        'minimum_matched_trials_per_team': args.min_trials_per_team,
        'valid_run_count': len(run_rows),
        'rejected_runs': rejected_runs,
        'groups': group_audits,
        'complete_matched_groups': sum(
            group['complete'] for group in group_audits),
        'campaign_protocol_complete': campaign_protocol_complete,
        'claim_authorized': False,
        'claim_authorization_note': (
            'Campaign completeness is a protocol diagnostic only. It does not '
            'authorize a physical scaling claim without independent review.'),
    }
    (output_dir / 'campaign_audit.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report['campaign_protocol_complete'] else 2


if __name__ == '__main__':
    sys.exit(main())
