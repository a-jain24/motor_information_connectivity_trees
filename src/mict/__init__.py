"""mict — motor-system Chow-Liu tree vs functional connectivity engine.

The pipeline boundary is the ROI-timeseries matrix ``(T, n_ROI)``: everything in
this package downstream of that boundary (mutual_info, connectivity, chow_liu,
clustering, stats) is space-agnostic and shared across datasets and image spaces.
"""

__version__ = "0.1.0"
