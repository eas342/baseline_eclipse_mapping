# README for Prep LC from Sim

## Notes on an offset in the code

Mirage may have a bug. It treats the out-of-transit flux (where the lightcurve is 1.0) differently from the in-transit flux. This speeds up the calculations so that it doesn't have to re-create the noiseless seed image for out-of-transit points. However, there may be an offset in the code.

<img src="plots/specific_saves/hd189733b_transit_weirdness2.png"></img>

This figure shows the offset. For the transit model, I therefore include an offset. Really this problem should be fixed within mirage.

