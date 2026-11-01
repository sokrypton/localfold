# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepMind's `predicted_tm_score`, the one RFD4-Proteina's AF2 worker scores a chain and a chain pair with.

Copied from deepmind/alphafold `alphafold/common/confidence.py` (its `_softmax`,
`_calculate_bin_centers` and `predicted_tm_score`, unchanged): upstream's AF2 runtime image puts
DeepMind's AlphaFold first on its path, and calls this function with an `interface` argument that
ColabFold's own `confidence.py` - the one the local ColabFold environments carry - does not take.
tools/check-rfd4p-oracle.py hands it to upstream's `_chain_summary` in place of that module.
"""

from typing import Optional

import numpy as np


def _softmax(x: np.ndarray, axis: Optional[int] = None):
  x = np.asarray(x)
  x_max = np.max(x, axis=axis, keepdims=True)
  exp_x_shifted = np.exp(x - x_max)
  return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def _calculate_bin_centers(breaks: np.ndarray):
  """Gets the bin centers from the bin edges.

  Args:
    breaks: [num_bins - 1] the error bin edges.

  Returns:
    bin_centers: [num_bins] the error bin centers.
  """
  step = breaks[1] - breaks[0]

  # Add half-step to get the center
  bin_centers = breaks + step / 2
  # Add a catch-all bin at the end.
  bin_centers = np.concatenate([bin_centers, [bin_centers[-1] + step]], axis=0)
  return bin_centers


def predicted_tm_score(
    logits: np.ndarray,
    breaks: np.ndarray,
    residue_weights: Optional[np.ndarray] = None,
    asym_id: Optional[np.ndarray] = None,
    interface: bool = False,
) -> np.ndarray:
  """Computes predicted TM alignment or predicted interface TM alignment score.

  Args:
    logits: [num_res, num_res, num_bins] the logits output from
      PredictedAlignedErrorHead.
    breaks: [num_bins] the error bins.
    residue_weights: [num_res] the per residue weights to use for the
      expectation.
    asym_id: [num_res] the asymmetric unit ID - the chain ID. Only needed for
      ipTM calculation, i.e. when interface=True.
    interface: If True, interface predicted TM score is computed.

  Returns:
    ptm_score: The predicted TM alignment or the predicted iTM score.
  """

  # residue_weights has to be in [0, 1], but can be floating-point, i.e. the
  # exp. resolved head's probability.
  if residue_weights is None:
    residue_weights = np.ones(logits.shape[0])

  bin_centers = _calculate_bin_centers(breaks)

  num_res = int(np.sum(residue_weights))
  # Clip num_res to avoid negative/undefined d0.
  clipped_num_res = max(num_res, 19)

  # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
  # "Scoring function for automated assessment of protein structure template
  # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
  d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8

  # Convert logits to probs.
  probs = np.array(_softmax(logits, axis=-1))

  # TM-Score term for every bin.
  tm_per_bin = 1.0 / (1 + np.square(bin_centers) / np.square(d0))
  # E_distances tm(distance).
  predicted_tm_term = np.sum(probs * tm_per_bin, axis=-1)

  pair_mask = np.ones(shape=(num_res, num_res), dtype=bool)
  if interface:
    pair_mask *= asym_id[:, None] != asym_id[None, :]

  predicted_tm_term *= pair_mask

  pair_residue_weights = pair_mask * (
      residue_weights[None, :] * residue_weights[:, None]
  )
  normed_residue_mask = pair_residue_weights / (
      1e-8 + np.sum(pair_residue_weights, axis=-1, keepdims=True)
  )
  per_alignment = np.sum(predicted_tm_term * normed_residue_mask, axis=-1)
  return np.asarray(per_alignment[(per_alignment * residue_weights).argmax()])
