"""A calibration set from UniRef50, with the length distribution folds have.

    python3 tools/esmc/fetch_calibration.py --out uniref50-calibration.fasta

🔴 THE HEAD OF `uniref50.fasta.gz` IS SORTED BY LENGTH AND IS NOT A SAMPLE.
Streaming the first 400 MB of it - the obvious way to avoid a 13 GB download -
gives 81,789 sequences whose MEDIAN length is 6796 residues, against the 60-250
of anything anyone folds. A calibration set of titins would put the Hessian's
mass in the wrong place and there would be nothing on screen to say so.

UniProt's REST search takes the length filter directly and pages with a cursor,
so this asks for what it wants instead of taking what comes first.
"""
from __future__ import annotations

import argparse
import urllib.parse
import urllib.request

ENDPOINT = 'https://rest.uniprot.org/uniref/search'


def page(query, size, cursor=None):
    parameters = {'query': query, 'format': 'fasta', 'size': str(size)}
    if cursor:
        parameters['cursor'] = cursor
    request = urllib.request.Request(
        ENDPOINT + '?' + urllib.parse.urlencode(parameters),
        headers={'Accept': 'text/plain'})
    with urllib.request.urlopen(request) as response:
        text = response.read().decode()
        link = response.headers.get('Link', '')
    following = None
    if 'rel="next"' in link:
        following = urllib.parse.parse_qs(
            urllib.parse.urlparse(link[1:link.index('>')]).query).get(
                'cursor', [None])[0]
    return text, following


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='uniref50-calibration.fasta')
    parser.add_argument('--count', type=int, default=4000)
    parser.add_argument('--min-length', type=int, default=60)
    parser.add_argument('--max-length', type=int, default=400)
    parser.add_argument('--page', type=int, default=500)
    arguments = parser.parse_args()

    query = ('identity:0.5 AND length:[%d TO %d]'
             % (arguments.min_length, arguments.max_length))
    written, cursor, chunks = 0, None, []
    while written < arguments.count:
        text, cursor = page(query, min(arguments.page,
                                       arguments.count - written), cursor)
        if not text:
            break
        chunks.append(text)
        written += text.count('>')
        print('  %d sequences' % written, flush=True)
        if not cursor:
            break
    with open(arguments.out, 'w') as handle:
        handle.write(''.join(chunks))
    print('wrote %s  (%d sequences, %d-%d residues)'
          % (arguments.out, written, arguments.min_length, arguments.max_length))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
