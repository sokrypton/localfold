"""Dump ESMFold2's folding trunk, so a WebGPU port has something to be wrong against.

    .venv-esm/bin/python tools/esmc/dump-esmfold2-trunk.py --sequence-length 40

Writes `oracle-dumps/esmfold2-trunk-<n>.json`.

🔴 THE TRUNK IS DUMPED FROM z_init, NOT FROM A SEQUENCE. Everything upstream -
the featuriser, the relative position encoding, the language-model shim - is a
separate port with its own failure modes, and a trunk checked end-to-end would
be checking all of them at once. Given z_init and the pair mask, the trunk is a
pure function; that is what this records.

🔴 AND IT RECORDS EVERY LOOP, BECAUSE THE RECURRENCE IS WHERE A PORT DIVERGES
SLOWLY. `z = z_init + pair_loop_proj(z)` then 24 blocks, four times over - a
port that is slightly wrong in one block reads as slightly wrong after the
first loop and badly wrong after the fourth, and only the per-loop values say
which.

🔴 AND THE LOOP RUNS n_loops + 1 TIMES. The config says `num_loops: 3` and the
model runs four iterations. Reading that as three is a silent quarter less
trunk, which still folds.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

SEQUENCE = ('MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYN'
            'IQKESTLHLVLRLRGGMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGT')


def tolist(tensor):
    return np.asarray(tensor.detach().cpu(), np.float32).reshape(-1).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esmfold2', default='esmfold2-fast-600m')
    parser.add_argument('--sequence-length', type=int, default=40)
    parser.add_argument('--out', default=None)
    parser.add_argument('--esmc', default=None,
                        help="ESM-C checkpoint directory; supplies lm_hidden_states "
                             "so the LANGUAGE MODEL's term of z_init is exercised")
    parser.add_argument('--float32-attention', action='store_true',
                        help='the CONTROL: neutralise the atom attention\'s '
                             'unconditional bfloat16 downcast')
    arguments = parser.parse_args()

    torch.set_grad_enabled(False)
    if arguments.float32_attention:
        # 🔴 SWA3DRoPEAttention CASTS q, k AND v TO bfloat16 WHATEVER THE MODEL'S
        # DTYPE - `if q.dtype not in (float16, bfloat16): q, k, v = q.bfloat16()...`
        # - so a float32 port cannot agree with it below about 1e-4 and the
        # residual looks exactly like a convention bug. Neutralising the cast is
        # the control that tells the two apart. It is global and deliberately
        # crude: this arm exists to answer one question, not to fold.
        torch.Tensor.bfloat16 = lambda self: self
    torch.set_num_threads(8)
    from esm.models.esmfold2 import EsmFold2ExperimentalModel
    from esm.models.esmfold2.protein_utils import prepare_protein_features

    sequence = SEQUENCE[:arguments.sequence_length]
    model = EsmFold2ExperimentalModel.from_pretrained(
        str(ROOT / arguments.esmfold2), load_esmc=False).eval()

    captured = {'loops': [], 'projected': [], 'inputs': []}

    def after_projection(_module, _inputs, output):
        captured['projected'].append(output.detach().clone())

    def after_trunk(_module, inputs, output):
        # 🔴 THE INPUT AS WELL AS THE OUTPUT, so the check is `trunk(x) == y` and
        # needs neither z_init nor the recurrence. The first version recorded
        # only outputs and pair_loop_proj(0), which cannot reproduce loop zero:
        # z_init never appears in either.
        pair = output[0] if isinstance(output, (tuple, list)) else output
        captured['loops'].append(pair.detach().clone())
        captured['inputs'].append(inputs[0].detach().clone())

    # 🔴 BLOCK ZERO'S THREE SUB-MODULES TOO, because a whole-trunk check that
    # fails says only "somewhere in 24 blocks". One tri-mul is a second to
    # verify and the whole trunk is 77; the ladder matters more than the total.
    captured['modules'] = {}
    captured['conditioning'] = []

    def module_hook(label):
        # 🔴 KEYWORDS TOO, BECAUSE HALF OF THESE TAKE NO POSITIONAL ARGUMENT AT
        # ALL. `rel_pos` and `inputs_embedder` are called entirely by keyword,
        # so a hook reading `inputs[0]` raises IndexError on them - and the
        # tempting fix, skipping a module with no positional input, would
        # silently drop exactly the two modules that most need recording.
        def hook(_module, inputs, keywords, output):
            if label in captured['modules']:
                return                     # the first loop only
            arguments = {}
            for index, value in enumerate(inputs):
                if torch.is_tensor(value):
                    arguments[str(index)] = value.detach().clone()
            for name, value in (keywords or {}).items():
                if torch.is_tensor(value):
                    arguments[name] = value.detach().clone()
            first = next(iter(arguments.values())) if arguments else None
            captured['modules'][label] = {
                'input': first,
                'arguments': arguments,
                'output': (output[0] if isinstance(output, (tuple, list))
                           else output).detach().clone(),
            }
        return hook

    # 🔴 EVERY ADDEND OF z_init, NOT JUST z_init. It is a sum of five terms -
    # two projections of the atom encoder's output, a relative position
    # encoding, a token-bond encoding and the language model's pair - and each
    # is its own port. A checker that only sees the total can say the sum is
    # wrong and nothing else; these say which term.
    first = model.folding_trunk.blocks[0]
    handles = [model.pair_loop_proj.register_forward_hook(after_projection),
               model.folding_trunk.register_forward_hook(after_trunk),
               first.tri_mul_out.register_forward_hook(module_hook('tri_mul_out'), with_kwargs=True),
               first.tri_mul_in.register_forward_hook(module_hook('tri_mul_in'), with_kwargs=True),
               first.pair_transition.register_forward_hook(module_hook('pair_transition'), with_kwargs=True),
               model.inputs_embedder.register_forward_hook(module_hook('inputs_embedder'), with_kwargs=True),
               model.z_init_1.register_forward_hook(module_hook('z_init_1'), with_kwargs=True),
               model.z_init_2.register_forward_hook(module_hook('z_init_2'), with_kwargs=True),
               model.rel_pos.register_forward_hook(module_hook('rel_pos'), with_kwargs=True),
               model.token_bonds.register_forward_hook(module_hook('token_bonds'), with_kwargs=True),
               model.language_model.register_forward_hook(module_hook('language_model'), with_kwargs=True)]

    # 🔴 AND INSIDE THE ATOM ENCODER, BECAUSE ITS OUTPUT ALONE SAYS "SOMEWHERE
    # IN THREE BLOCKS AND A POOLING". The whole-module check read 2e-4 with
    # every primitive verified by reading, which is exactly when a ladder is
    # worth more than another re-reading.
    encoder = model.inputs_embedder.atom_attention_encoder
    for label, module in (('atom.linear', encoder.atom_linear),
                          ('atom.norm', encoder.atom_norm),
                          ('atom.toToken', encoder.atom_to_token_linear)):
        handles.append(module.register_forward_hook(module_hook(label), with_kwargs=True))
    # 🔴 AND THE DIFFUSION CONDITIONING, whose FIRST call is the one recorded.
    # The sampler runs fifteen steps and caches `z` across them while `s`
    # depends on the noise level, so a hook that kept the last call would record
    # a different t_hat from the one it also recorded as an argument.
    # 🔴 AND IT RETURNS A TUPLE (s, z), SO module_hook WOULD DROP HALF OF IT.
    # That helper keeps `output[0]` for the modules that wrap their answer in a
    # tuple; here both elements ARE the answer, and recording only `s` would
    # leave the pair conditioning - the larger and more easily wrong half -
    # unchecked while every check passed.
    diffusion = model.structure_head.diffusion_module

    def conditioning_hook(_module, inputs, keywords, output):
        if captured['conditioning']:
            return                                    # the first step only
        single, pair = output
        arguments = {}
        for index, value in enumerate(inputs):
            if torch.is_tensor(value):
                arguments[str(index)] = value.detach().clone()
        for name, value in (keywords or {}).items():
            if torch.is_tensor(value):
                arguments[name] = value.detach().clone()
        captured['conditioning'].append({
            'single': single.detach().clone(), 'pair': pair.detach().clone(),
            'arguments': arguments,
        })

    handles.append(diffusion.conditioning.register_forward_hook(
        conditioning_hook, with_kwargs=True))

    # The token transformer, whose twelve blocks are the denoiser's bulk. Its
    # output IS `output[0]` - it returns (a, intermediates) - so the shared
    # helper is right here where it was wrong for the conditioning.
    handles.append(diffusion.token_transformer.register_forward_hook(
        module_hook('diffusion.tokenTransformer'), with_kwargs=True))

    for index, block in enumerate(encoder.atom_transformer.blocks):
        handles.append(block.register_forward_hook(
            module_hook('atom.block%d' % index), with_kwargs=True))
        handles.append(block.attn.register_forward_hook(
            module_hook('atom.block%d.attn' % index), with_kwargs=True))

    features = prepare_protein_features(sequence)
    # 🔴 THE HIDDEN STATES ARE INJECTED, NOT LOADED. `esmc_id` names a separate
    # 2.3 GB artefact this checkpoint does not carry, and `load_esmc=True` would
    # fetch it - while the forward already accepts `lm_hidden_states` directly.
    # tools/esmc/esmc_forward.py's tower is the one this repository has checked
    # against transformers (2.1e-6) and against `esm`'s own language model
    # (3.2e-7), so injecting from it also makes the ESM-C half of the pipeline
    # the SAME code the WebGPU port is checked against.
    lm_hidden_states = None
    if arguments.esmc is not None:
        from esmc_forward import Checkpoint, Tower
        tower = Tower(Checkpoint(arguments.esmc))
        ids = np.asarray(features['input_ids'].detach().cpu()).reshape(-1)
        states = tower.hidden_states(ids)                 # (layers + 1, L, d_model)
        lm_hidden_states = states.permute(1, 0, 2).unsqueeze(0)   # (1, L, layers+1, d)
        print('  lm_hidden_states %s from %s'
              % (list(lm_hidden_states.shape), arguments.esmc))
    # 🔴 AND THE FEATURES THEMSELVES, because rel_pos is a pure function of five
    # integer arrays and the atom encoder of seven. A JavaScript featuriser that
    # builds them differently is a fault this dump can localise only if it
    # records what the model was actually given.
    captured['features'] = {}
    for name in ('residue_index', 'asym_id', 'sym_id', 'entity_id', 'token_index',
                 'token_bonds', 'mol_type', 'atom_to_token', 'ref_pos',
                 'ref_space_uid', 'ref_charge', 'ref_element',
                 'ref_atom_name_chars', 'atom_attention_mask',
                 'token_attention_mask', 'input_ids'):
        value = features.get(name)
        if value is None:
            continue
        captured['features'][name] = {
            'shape': list(value.shape),
            'values': np.asarray(value.detach().cpu()).reshape(-1).tolist(),
        }
    # 🔴 NO LANGUAGE MODEL, DELIBERATELY. lm_z is added to z_init once and is
    # this port's OTHER half; leaving it out makes the trunk's own arithmetic
    # the only thing recorded, and the tower already has its own oracle.
    if lm_hidden_states is not None:
        features['lm_hidden_states'] = lm_hidden_states
    output = model(**features, num_diffusion_samples=1, seed=0)
    for handle in handles:
        handle.remove()

    if not captured['loops']:
        raise SystemExit('the trunk hook never fired; the module name has moved')

    trunk = model.folding_trunk
    blocks = len(trunk.blocks) if hasattr(trunk, 'blocks') else None
    payload = {
        'sequence': sequence,
        'esmfold2': arguments.esmfold2,
        'loops': len(captured['loops']),
        # 🔴 WHICH ARM THIS IS, IN THE ARTEFACT. A port that agrees with the
        # control to 7e-8 and with the shipping model to 2e-4 is CORRECT, and a
        # checker handed the wrong dump with the wrong bound says the opposite.
        'float32Attention': bool(arguments.float32_attention),
        'languageModel': arguments.esmc,
        'blocks': blocks,
        'shapes': {'pair': list(captured['loops'][0].shape)},
        # The first projection sees a zero z, so its output is
        # pair_loop_proj(0) - the bias path alone - and z_init is what the
        # model added to it. Both are recorded rather than reconstructed.
        'block0': {name: {
            'input': tolist(v['input']) if v['input'] is not None else None,
            'arguments': {k: {'shape': list(t.shape), 'values': tolist(t)}
                          for k, t in v['arguments'].items()},
            'output': tolist(v['output']),
            'outputShape': list(v['output'].shape)}
                   for name, v in captured['modules'].items()},
        'features': captured['features'],
        'intoLoop': {str(i): tolist(v) for i, v in enumerate(captured['inputs'])},
        'afterLoop': {str(i): tolist(v) for i, v in enumerate(captured['loops'])},
        'distogram': tolist(output['distogram_logits'])
        if 'distogram_logits' in output else None,
        # The sampler's answer, which is the only thing a structure port can be
        # finally wrong against.
        'coordinates': tolist(output['sample_atom_coords'])
        if 'sample_atom_coords' in output else None,
        'conditioning': {
            'single': tolist(captured['conditioning'][0]['single']),
            'singleShape': list(captured['conditioning'][0]['single'].shape),
            'pair': tolist(captured['conditioning'][0]['pair']),
            'pairShape': list(captured['conditioning'][0]['pair'].shape),
            'arguments': {k: {'shape': list(t.shape), 'values': tolist(t)}
                          for k, t in captured['conditioning'][0]['arguments'].items()},
        } if captured['conditioning'] else None,
    }

    out = pathlib.Path(arguments.out) if arguments.out else (
        ROOT / 'oracle-dumps' / ('esmfold2-trunk-%d.json' % len(sequence)))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    print('wrote %s  (%d residues, %d loops, %s blocks, %.1f MB)'
          % (out, len(sequence), len(captured['loops']), blocks,
             out.stat().st_size / 1e6))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
