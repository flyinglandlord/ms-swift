"""Explicit segment schema; no silent truncation or implicit train/test splitting."""
import json
import torch
from pathlib import Path


def read_records(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty dataset')
    seen = set()
    for row in rows:
        if row['id'] in seen:
            raise ValueError('Duplicate example id')
        seen.add(row['id'])
        for field in ('segments', 'query_ids', 'answer_ids'):
            if not row.get(field):
                raise ValueError(f'Empty {field} in {row["id"]}')
        flat = (
            row.get('system_ids', []) + row.get('cache_prefix_ids', []) + sum(row['segments'], []) + row['query_ids']
            + row['answer_ids'])
        if any(not isinstance(i, int) or i < 0 for i in flat) or any(not s for s in row['segments']):
            raise ValueError('Token IDs must be nonnegative; segments must be nonempty')
    return rows


def assert_disjoint(train, validation):

    def keys(rows):
        return {r.get('trace_id', r['id']) for r in rows}

    if keys(train) & keys(validation):
        raise ValueError('Train/validation trajectory leakage')


class SegmentCollator:
    """Batch size one avoids fictitious empty segments in the upstream model."""

    def __call__(self, rows):
        if len(rows) != 1:
            raise ValueError('PIC v1 supports per-device batch size 1; use gradient accumulation')
        row = rows[0]
        docs = row['segments']
        width = max(map(len, docs))
        ids = row['query_ids'] + row['answer_ids']
        return {
            'cache_prefix_ids': torch.tensor([row.get('cache_prefix_ids', [])], dtype=torch.long),
            'system_input_ids': torch.tensor([row.get('system_ids', [])], dtype=torch.long),
            'context_input_ids': torch.tensor([[x + [-100] * (width - len(x)) for x in docs]], dtype=torch.long),
            'input_ids': torch.tensor([ids]),
            'attention_mask': torch.ones(1, len(ids), dtype=torch.bool),
            'labels': torch.tensor([[-100] * len(row['query_ids']) + row['answer_ids']]),
        }
