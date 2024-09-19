# Grid-less-DOA
Sequential DOA Trajectory Estimation using Deep Complex Network and Residual Signals

## Abstract:
We propose a data-driven method for direction-of-arrival (DOA) trajectory estimation. We use a deep complex architecture which leverages complex-valued representations to capture both magnitude and phase information in the received sensor array data. The network is designed to output the DOA trajectory parameters and amplitudes of the strongest source. Deviating from conventional methods, which attempt to estimate parameters for all sources simultaneously -- leading to assignment ambiguity and the problem of uncertain output dimensions, we adopt a sequential approach. The estimated source signal contribution is subtracted from the input to obtain a residual signal. This residual signal is then fed back into the network to identify the next strongest source and so on, making the proposed network reusable. We evaluate our network on simulated data of varying complexity. Results demonstrate the feasibility of such a reusable network and potential improvements can be explored in future.

## Signal Model:
