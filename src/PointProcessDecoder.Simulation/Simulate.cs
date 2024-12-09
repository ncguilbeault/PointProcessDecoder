using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;
using TorchSharp;

namespace PointProcessDecoder.Simulation;

public static class Simulate
{
    public static Tensor Position1D(int steps, int cycles, double min, double max, ScalarType? scalarType = null)
    {
        if (scalarType != null) return ((sin(linspace(0, 2.0 * Math.PI * cycles, steps * cycles)) + 1.0) * 0.5 * (max - min) + min).to_type(scalarType.Value);
        return (sin(linspace(0, max * cycles, steps * cycles)) + 1.0) * 0.5 * (max - min) + min;
    }

    public static Tensor Position2D(int steps, int cycles, double xMin, double xMax, double yMin, double yMax, double scale, ScalarType? scalarType = null)
    {
        using (var _ = NewDisposeScope())
        {
            var x = linspace(xMin, xMax, steps);
            for (int i = 1; i < cycles; i++)
            {
                if (i % 2 == 0)
                {
                    x = hstack(x, linspace(xMin, xMax, steps));
                }
                else
                {
                    x = hstack(x, linspace(xMax, 0, steps));
                }
            }
            var y = (sin(linspace(0, yMax * cycles, steps * cycles) * scale) + 1.0) * 0.5 * (yMax - yMin) + yMin;
            var positionData = vstack([ x, y ]).T;
            if (scalarType != null) positionData = positionData.to_type(scalarType.Value);
            return positionData.MoveToOuterDisposeScope();
        }
    }

    public static Tensor PlaceFields(double xMin, double xMax, int numNeurons, int? seed = null, ScalarType? scalarType = null)
    {
        using (var _ = NewDisposeScope())
        {
            if (seed != null) manual_seed(seed.Value);
            var positions = linspace(xMin + 0.5 * (xMax - xMin) / numNeurons, xMax - 0.5 * (xMax - xMin) / numNeurons, numNeurons);
            if (scalarType != null) positions = positions.to_type(scalarType.Value);
            return positions.MoveToOuterDisposeScope();
        }
    }

    public static Tensor PlaceFields(double xMin, double xMax, double yMin, double yMax, int numNeurons, int? seed = null, ScalarType? scalarType = null)
    {
        using (var _ = NewDisposeScope())
        {
            if (seed != null) manual_seed(seed.Value);
            var count = sqrt(numNeurons).to_type(ScalarType.Int32).item<int>();
            var xSpacing = linspace(xMin + 0.5 * (xMax - xMin) / count, xMax - 0.5 * (xMax - xMin) / count, count);
            var ySpacing = linspace(yMin + 0.5 * (yMax - yMin) / count, yMax - 0.5 * (yMax - yMin) / count, count);
            var positionGrid = meshgrid([ xSpacing, ySpacing ]);
            var centers = vstack([ positionGrid[0].flatten(), positionGrid[1].flatten() ]).T;
            var leftover = rand(numNeurons - count * count, 2) * tensor(new double[] { xMax - xMin, yMax - yMin }) + tensor(new double[] { xMin, yMin });
            var positions = vstack([ centers, leftover ]);
            if (scalarType != null) positions = positions.to_type(scalarType.Value);
            return positions.MoveToOuterDisposeScope();
        }
    }

    public static Tensor GaussianSpikeRates(Tensor pos, Tensor pos0, Tensor sigma, ScalarType? scalarType = null)
    {
        using (var _ = NewDisposeScope())
        {
            // var spikingRates = exp(-((x - x0).pow(2) + (y - y0).pow(2)) / (2.0 * sigma.pow(2)));
            var posDiff = pos.unsqueeze(1) - pos0;
            var diffSquared = posDiff.pow(2);
            var diffSquaredSum = diffSquared.sum(dim: -1);
            var sigmaSquared = sigma.pow(2);
            var diffOverSigma = diffSquaredSum / (2.0 * sigmaSquared);
            var spikingRates = exp(-diffOverSigma);
            if (scalarType != null) spikingRates = spikingRates.to_type(scalarType.Value);
            return spikingRates.MoveToOuterDisposeScope();
        }
    }

    public static Tensor SpikesAtPosition(Tensor positionData, Tensor placeFieldCenters, Tensor placeFieldRadius, Tensor firingThreshold, int? seed = null, ScalarType? scalarType = null, double? noiseScale = null)
    {
        using (var _ = NewDisposeScope())
        {
            if (seed != null) manual_seed(seed.Value);
            var spikingRates = GaussianSpikeRates(positionData, placeFieldCenters, placeFieldRadius, scalarType);
            var noise = rand_like(spikingRates);
            noise *= noiseScale ?? 1.0;
            var spikeThreshold = firingThreshold + noise;
            var spikes = spikeThreshold < spikingRates;
            if (scalarType != null) spikes = spikes.to_type(scalarType.Value);
            return spikes.T.MoveToOuterDisposeScope();
        }
    }
}