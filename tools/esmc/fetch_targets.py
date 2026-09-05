"""A held-out target set: single protein chains released after the data cutoff.

    python3 tools/esmc/fetch_targets.py --out targets --count 16

🔴 THE `-step` CHECKPOINTS HAVE A SEPTEMBER 2021 CUTOFF, SO A TARGET FROM THE
PDB IS ONLY A TEST IF IT WAS DEPOSITED AFTER IT. Three fixtures that this
repository has scored AF3 against for months are exactly the wrong set to
generalise a language model's compression from - they are old, small, and two of
the three are de novo designs. This asks the RCSB for entries the model cannot
have seen.

The filters are deliberately narrow: X-ray, one polymer entity, one chain in the
asymmetric unit, 60-220 residues, better than 2.0 A. That is not a benchmark, it
is a set of targets where a fold either works or does not and nothing else is in
the way.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import urllib.parse
import urllib.request

SEARCH = 'https://search.rcsb.org/rcsbsearch/v2/query?json='
DOWNLOAD = 'https://files.rcsb.org/download/%s.pdb'


def query(count, after, low, high, resolution):
    return {
        'query': {'type': 'group', 'logical_operator': 'and', 'nodes': [
            {'type': 'terminal', 'service': 'text', 'parameters': {
                'attribute': 'rcsb_accession_info.initial_release_date',
                'operator': 'greater', 'value': after}},
            {'type': 'terminal', 'service': 'text', 'parameters': {
                'attribute': 'exptl.method',
                'operator': 'exact_match', 'value': 'X-RAY DIFFRACTION'}},
            {'type': 'terminal', 'service': 'text', 'parameters': {
                'attribute': 'rcsb_entry_info.resolution_combined',
                'operator': 'less', 'value': resolution}},
            {'type': 'terminal', 'service': 'text', 'parameters': {
                'attribute': 'rcsb_entry_info.polymer_entity_count_protein',
                'operator': 'equals', 'value': 1}},
            {'type': 'terminal', 'service': 'text', 'parameters': {
                'attribute': 'rcsb_entry_info.deposited_polymer_entity_instance_count',
                'operator': 'equals', 'value': 1}},
            {'type': 'terminal', 'service': 'text', 'parameters': {
                'attribute': 'rcsb_entry_info.deposited_polymer_monomer_count',
                'operator': 'range',
                'value': {'from': low, 'to': high, 'include_lower': True,
                          'include_upper': True}}},
        ]},
        'return_type': 'entry',
        'request_options': {'paginate': {'start': 0, 'rows': count},
                            'sort': [{'sort_by': 'rcsb_accession_info.initial_release_date',
                                      'direction': 'desc'}]},
    }


THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'MSE': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T',
    'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


def chain_sequence(text):
    """The first chain's resolved residues, by the same rule the probe uses."""
    seen, out, chain = set(), [], None
    for line in text.splitlines():
        if line[:4] not in ('ATOM', 'HETA') or line[12:16].strip() != 'CA':
            continue
        if chain is None:
            chain = line[21]
        if line[21] != chain:
            break
        key = line[22:27]
        if key in seen:
            continue
        seen.add(key)
        out.append(THREE_TO_ONE.get(line[17:20].strip(), 'X'))
    return ''.join(out)


def similar(a, b, threshold=0.6):
    """Crude ungapped identity - enough to catch a deposition group."""
    if not a or not b:
        return False
    if abs(len(a) - len(b)) > 0.3 * max(len(a), len(b)):
        return False
    n = min(len(a), len(b))
    return sum(x == y for x, y in zip(a[:n], b[:n])) / n >= threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='targets')
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--pool', type=int, default=200,
                        help='entries to consider before the identity filter')
    parser.add_argument('--after', default='2022-01-01')
    parser.add_argument('--min-residues', type=int, default=60)
    parser.add_argument('--max-residues', type=int, default=220)
    parser.add_argument('--resolution', type=float, default=2.0)
    arguments = parser.parse_args()

    body = query(arguments.pool, arguments.after, arguments.min_residues,
                 arguments.max_residues, arguments.resolution)
    url = SEARCH + urllib.parse.quote(json.dumps(body))
    with urllib.request.urlopen(url) as response:
        found = json.load(response)
    ids = [row['identifier'] for row in found['result_set']]
    print('%d entries released after %s' % (len(ids), arguments.after))

    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    # 🔴 NOT EVERY ENTRY HAS A PDB-FORMAT FILE. The RCSB stopped writing one for
    # entries that do not fit the format, and asking for it answers 404 - which
    # is a property of that entry and not an error in the run, so it is skipped
    # and counted rather than raised.
    kept, missing, near, sequences = [], [], 0, []
    for code in ids:
        if len(kept) >= arguments.count:
            break
        path = out / ('%s.pdb' % code.lower())
        if not path.exists():
            try:
                with urllib.request.urlopen(DOWNLOAD % code) as response:
                    path.write_bytes(response.read())
            except urllib.error.HTTPError as error:
                missing.append('%s (%d)' % (code, error.code))
                continue
        sequence = chain_sequence(path.read_text())
        if not arguments.min_residues <= len(sequence) <= arguments.max_residues:
            path.unlink()
            continue
        # 🔴 THE RCSB RETURNS DEPOSITION GROUPS TOGETHER, so a plain "newest 20"
        # came back as nine consecutive entries of the same protein. Filtering
        # on identity is not a nicety here: nine copies of one target read as
        # nine independent measurements and are one.
        if any(similar(sequence, seen) for seen in sequences):
            path.unlink()
            near += 1
            continue
        sequences.append(sequence)
        kept.append((code, len(sequence)))

    print('%d kept in %s' % (len(kept), out))
    for code, length in kept:
        print('  %s  %d residues' % (code.lower(), length))
    if near:
        print('%d dropped as near-duplicates of one already kept' % near)
    if missing:
        print('no PDB-format file for: %s' % ', '.join(missing))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
