"""Publish traceable evaluation summaries and minimum-edit error examples."""
import csv
import json
import math
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from .alignment import align
from .wer import corpus_wer, edit_counts
from ..core.settings import read_json, write_json
from ..core.provenance import file_sha256


def summarize_evaluation(path):
    report = read_json(path)
    predictions = report.get('predictions', [])
    if not predictions:
        raise ValueError(f'Evaluation has no predictions: {path}')
    ids = set()
    errors, confusions = [], Counter()
    for row in predictions:
        if not isinstance(row.get('id'), str) or row['id'] in ids:
            raise ValueError(f'Evaluation requires unique clip IDs: {path}')
        ids.add(row['id'])
        for field in ('reference', 'prediction'):
            if not isinstance(row.get(field), list) or not all(isinstance(g, str) for g in row[field]):
                raise ValueError(f'Evaluation requires lists of gloss strings: {path}')
        counts = edit_counts(row['reference'], row['prediction'])
        pairs = align(row['reference'], row['prediction'])
        confusions.update((target, predicted) for target, predicted in pairs if target != predicted)
        errors.append({**row, **counts, 'alignment': pairs})
    metrics = corpus_wer([r['reference'] for r in predictions], [r['prediction'] for r in predictions])
    for key, value in metrics.items():
        recorded = report.get('metrics', {}).get(key)
        if recorded is None and value is None:
            continue
        if not isinstance(recorded, (int, float)) or value is None or not math.isclose(recorded, value, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f'Recorded {key} differs from predictions: {path}')
    synthetic = report.get('synthetic_debug', report.get('synthetic_debug_only'))
    status = 'synthetic diagnostic' if synthetic is True else 'measured evaluation'
    if synthetic is None:
        status = 'data origin unspecified'
    if report.get('subset'):
        status += ', subset'
    return {'source': str(Path(path).resolve()), 'source_sha256': file_sha256(path),
            'head': report.get('head', report.get('model', 'unspecified')),
            'split': report.get('split', 'unspecified'), 'status': status, 'metrics': metrics,
            'metadata': {k: v for k, v in report.items() if k not in {'predictions', 'metrics'}},
            'correct_examples': [row for row in errors if row['errors'] == 0][:5],
            'error_examples': sorted((row for row in errors if row['errors']),
                                     key=lambda row: row['errors'], reverse=True)[:10],
            'confusions': [{'reference': pair[0], 'prediction': pair[1], 'count': count}
                           for pair, count in confusions.most_common(20)]}


def final_report(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('Report directory exists; choose a new output')
    paths = [Path(path).resolve() for path in args.evaluations]
    if len(set(paths)) != len(paths):
        raise ValueError('Evaluation inputs must be unique')
    results = [summarize_evaluation(path) for path in paths]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.phase5-', dir=output.parent) as directory:
        staged = Path(directory) / 'report'
        staged.mkdir()
        write_json(staged / 'results.json', {'evaluations': results})
        with (staged / 'results.csv').open('w', encoding='utf-8', newline='') as stream:
            fields = ['source', 'head', 'split', 'status', 'videos', 'wer_percent',
                      'substitutions', 'deletions', 'insertions', 'reference_words']
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            for result in results:
                writer.writerow({**result, **result['metrics']})
        paragraphs = ['# Evaluation report',
                      'Each result retains its source file, checkpoint identity, decoder and evaluation metadata in results.json.',
                      'Interpret comparisons using matching datasets, candidate labels and scoring rules. Use compare-runs to validate training protocols and repeated seeds.']
        for result in results:
            metrics = result['metrics']
            paragraphs.extend([f"## {result['head']} ({result['split']})",
                               f"Evidence status — {result['status']}. Clips scored — {metrics['videos']}. WER — {metrics['wer_percent']}%.",
                               f"Substitutions — {metrics['substitutions']}. Missing glosses — {metrics['deletions']}. Extra glosses — {metrics['insertions']}.",
                               f"Source file — {result['source']}"])
            for row in result['correct_examples'] + result['error_examples']:
                # JSON quoting keeps arbitrary labels and IDs out of Markdown structure.
                paragraphs.append('```json\n' + json.dumps(row, ensure_ascii=False, indent=2) + '\n```')
        (staged / 'report.md').write_text('\n\n'.join(paragraphs) + '\n', encoding='utf-8')
        staged.rename(output)
    print(f'Final report saved: {output}')
