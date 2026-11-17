# wedgefill

Complete cryo-ET membrane segmentations where the missing wedge hid them.

In a single-axis tilt series (about ±60°), the missing wedge removes the signal of membrane whose normal is
close to the beam axis. Membrane segmentations such as MemBrain-seg therefore come out as open "( )" arcs in XZ
slices: the tops and bottoms of vesicles and cells are missing. Downstream tools trim these open surfaces
(e.g. surface morphometrics) or fill the missing wedge in the density without a membrane model (IsoNet,
DeepDeWedge). wedgefill completes the **segmentation** itself, and only where the wedge could have hidden
membrane.

## Method

For each membrane object, a closed triangle mesh is fitted to the mask voxels with:

- a robust data term: every mask voxel lies near the surface;
- a **visible-direction constraint**: surface whose normal points more than `hidden_deg` from the beam axis must
  lie on the mask (it is never invented);
- a **Willmore (bending-energy) prior** that shapes the rest.

Only off-mask surface whose normal lies inside the hidden cone is returned as completion. Openings visible in XY
stay open, and concave shapes are not bridged.

```python
import numpy as np
from wedgefill import complete_membrane

points = np.argwhere(mask_of_one_membrane)            # (z, y, x); z = beam axis
res = complete_membrane(points, hidden_deg=55)
res.vertices, res.faces, res.added                    # mesh + faces added as completion
```

`scripts/complete_tomogram.py` runs this on objects of a tomogram + segmentation (Zarr) and saves slice figures.

## Status: research prototype

- **Real data:** tested on CZ cryoET Data Portal dataset 10235 (Salmonella minicells), with MemBrain-seg masks.
  - The gaps are real and wedge-shaped. MemBrain loses membrane with normals up to about 55° from the beam
    axis, wider than the geometric wedge.
  - The completed caps are plausible, and on one double-membrane minicell the outer cap height agrees with an
    independent estimate.
- **Not validated:** the hidden region contains no membrane signal, so completed shapes are the prior's
  prediction, not a measurement.
  - Curvature or height in completed regions should be reported with that caveat.
  - Validation against ground truth (dual-axis tomograms, or phantoms with realistic tilt-series simulation) is
    the next step.
- **Known bias:** the Willmore prior favours round shapes. A Helfrich model with volume or pressure terms is a
  planned extension.

## Install and test

```bash
pip install -e ".[scripts,test]"
pytest
```

## License

MIT
