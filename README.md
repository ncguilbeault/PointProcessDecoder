# PointProcessDecoder

This repo contains a C# implementation of the Bayesian state space point process neural decoder. The code is based on the TorchSharp library for .NET/C# and is inspired by the [replay_trajectory_classification repository](https://github.com/Eden-Kramer-Lab/replay_trajectory_classification) from the Eden-Kramer Lab. It provides a flexible framework for performing neural decoding of observations from spike-train or clusterless mark data.

## Overview

The goal of this software is to perform neural decoding. Bayesian state-space models, in particular, provide a framework to model the transitions between states based on neural activity and point processes capture the probabilistic relationship between neural activity and observations.

## Features

* Flexible - many components of the model support custom or user-defined classes with the appropriate interface.
* TorchSharp integration - supports both CPU and GPU-acceleration

## Steps to Build

1. Install .NET 8:
Download [the .NET SDK](https://dotnet.microsoft.com/download) if you haven't already.

2. Clone the repository:

```cmd
git clone https://github.com/ncguilbeault/PointProcessDecoder.cs
cd PointProcessDecoder
```

3. Restore dependencies:

```cmd
dotnet restore
```

4. Build the solution:

```
dotnet build
```

## Quickstart

Here is a minimal example of how to use the decoder in a console app:

```csharp
using PointProcessDecoder.Core;
using PointProcessDecoder.Plot;
using PointProcessDecoder.Simulation;

namespace DecoderDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            // 1. Load data.
            // Example: Generate simulated data
            (position, spikeCounts) = Simulation.Utilities.InitializeSimulation1D(
                numNeurons: 40,
                placeFieldRadius: 0.8,
                firingThreshold: 0.2
            );

            // 2. Create the model and select parameters.
            var model = new PointProcessModel(
                estimationMethod: Core.Estimation.EstimationMethod.KernelDensity,
                transitionsType: Core.Transitions.TransitionsType.Uniform,
                encoderType: Core.Encoder.EncoderType.SortedSpikeEncoder,
                decoderType: Core.Decoder.DecoderType.StateSpaceDecoder,
                stateSpaceType: Core.StateSpace.StateSpaceType.DiscreteUniformStateSpace,
                likelihoodType: Core.Likelihood.LikelihoodType.Poisson,
                minStateSpace: [0],
                maxStateSpace: [120],
                stepsStateSpace: [50],
                observationBandwidth: [5],
                stateSpaceDimensions: 1,
                nUnits: 40
            );

            // 4. Encode neural data and observations
            model.Encode(spikeCounts, position);

            // 5. Predict or decode observations from spikes
            var prediction = model.Decode(spikeCounts);

            // 6. Display results
            Heatmap plotPrediction = new(
                xMin: 0,
                xMax: steps * cycles,
                yMin: 0,
                yMax: 120,
                title: "Prediction"
            );

            plotPrediction.Show<float>(
                prediction
            );

            plotPrediction.Save(png: true);
        }
    }
}
```

## References

This work is based on several previously published works. If you use this software, consider citing the following:

1. Denovellis, E. L., Gillespie, A. K., Coulter, M. E., Sosa, M., Chung, J. E., Eden, U. T., & Frank, L. M. (2021). Hippocampal replay of experience at real-world speeds. Elife, 10, e64505.

2. Sodkomkham, D., Ciliberti, D., Wilson, M. A., Fukui, K. I., Moriyama, K., Numao, M., & Kloosterman, F. (2016). Kernel density compression for real-time Bayesian encoding/decoding of unsorted hippocampal spikes. Knowledge-Based Systems, 94, 1-12.


**Contributions and feedback are welcome!**