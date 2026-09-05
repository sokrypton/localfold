"""Download an ESM-C tower and its ESMFold2 folding model from Hugging Face.

    python3 tools/esmc/fetch.py            # the 600M pair, 3.0 GB
    python3 tools/esmc/fetch.py --size 300m

Neither is gated and both are MIT, unlike DeepMind's AF3 parameters - so there
is no terms dialog to answer and nothing build_site.py has to refuse.

🔴 THE TOWER MUST MATCH THE FOLDING MODEL'S `esmc_id`, NOT JUST ITS WIDTH. The
folding model's layer mix is a (n_layers + 1,) vector trained against ONE
checkpoint of the tower, so pairing `ESMFold2-Experimental-Fast-base600M-step1500k`
with the general-release `esmc-600m-2024-12` gives 37 weights over 37 states of a
different model. The pairing is read out of the folding model's own config here
rather than written down twice.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

FOLDING = {
    '600m': 'biohub/ESMFold2-Experimental-Fast-base600M-step1500k',
    '300m': 'biohub/ESMFold2-Experimental-Fast-base300M-step1500k',
    '6b': 'biohub/ESMFold2-Experimental-Fast',
}


def config_of(repo):
    url = 'https://huggingface.co/%s/raw/main/config.json' % repo
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default='600m', choices=sorted(FOLDING))
    parser.add_argument('--out', default=None,
                        help='defaults to esmc-<size>/ and esmfold2-fast-<size>/')
    arguments = parser.parse_args()

    from huggingface_hub import snapshot_download

    folding = FOLDING[arguments.size]
    tower = config_of(folding)['esmc_id']
    print('%s folds from %s' % (folding, tower))
    pairs = [(tower, ROOT / ('esmc-%s' % arguments.size)),
             (folding, ROOT / ('esmfold2-fast-%s' % arguments.size))]
    for repo, destination in pairs:
        print('-> %s' % destination)
        snapshot_download(repo, local_dir=str(destination))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
