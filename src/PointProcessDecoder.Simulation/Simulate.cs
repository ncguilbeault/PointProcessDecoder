using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using static TorchSharp.torch;
using TorchSharp;

namespace PointProcessDecoder.Simulation;

/// <summary>
/// Class to simulate data for testing.
/// </summary>
public static class Simulate
{
    /// <summary>
    /// Simulate a 1D position using a sine wave with a given number of steps and cycles, scaling the output to the given min and max values.
    /// </summary>
    /// <param name="steps"></param>
    /// <param name="cycles"></param>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <param name="scalarType"></param>
    /// <returns>
    /// A 1D tensor of position values with shape (steps * cycles, 1).
    /// </returns>
    public static Tensor Position(int steps, int cycles, double min, double max, ScalarType? scalarType = null)
    {
        if (scalarType != null) return ((sin(linspace(0, 2.0 * Math.PI * cycles, steps * cycles)) + 1.0) * 0.5 * (max - min) + min).to_type(scalarType.Value).unsqueeze(1);
        return ((sin(linspace(0, max * cycles, steps * cycles)) + 1.0) * 0.5 * (max - min) + min).unsqueeze(1);
    }

    /// <summary>
    /// Simulate a 2D position using a sine wave with a given number of steps and cycles, scaling the output to the given min and max values.
    /// At each cycle, the x values are reversed.
    /// </summary>
    /// <param name="steps"></param>
    /// <param name="cycles"></param>
    /// <param name="xMin"></param>
    /// <param name="xMax"></param>
    /// <param name="yMin"></param>
    /// <param name="yMax"></param>
    /// <param name="scale"></param>
    /// <param name="scalarType"></param>
    /// <returns>
    /// A 2D tensor of position values with shape (steps * cycles, 2).
    /// </returns>
    public static Tensor Position(int steps, int cycles, double xMin, double xMax, double yMin, double yMax, double scale, ScalarType? scalarType = null)
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

    /// <summary>
    /// Simulate 1D place fields for a given range of x values and number of neurons.
    /// Spaces the place fields evenly across the range.
    /// </summary>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <param name="numNeurons"></param>
    /// <param name="seed"></param>
    /// <param name="scalarType"></param>
    /// <returns>
    /// A 1D tensor of place field centers with shape (numNeurons).
    /// </returns>
    public static Tensor PlaceFieldCenters(double min, double max, int numNeurons, int? seed = null, ScalarType? scalarType = null)
    {
        using (var _ = NewDisposeScope())
        {
            if (seed != null) manual_seed(seed.Value);
            var positions = linspace(min + 0.5 * (max - min) / numNeurons, max - 0.5 * (max - min) / numNeurons, numNeurons);
            if (scalarType != null) positions = positions.to_type(scalarType.Value);
            return positions.MoveToOuterDisposeScope();
        }
    }

    /// <summary>
    /// Simulate 2D place fields for a given range of x and y values and number of neurons.
    /// Spaces the place fields evenly across the range for both x and y.
    /// Any leftover neurons are placed randomly within the range.
    /// </summary>
    /// <param name="xMin"></param>
    /// <param name="xMax"></param>
    /// <param name="yMin"></param>
    /// <param name="yMax"></param>
    /// <param name="numNeurons"></param>
    /// <param name="seed"></param>
    /// <param name="scalarType"></param>
    /// <returns>
    /// A 2D tensor of place field centers with shape (numNeurons, 2).
    /// </returns>
    public static Tensor PlaceFieldCenters(double xMin, double xMax, double yMin, double yMax, int numNeurons, int? seed = null, ScalarType? scalarType = null)
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

    /// <summary>
    /// Simulate Gaussian spike rates for a given position, center, and sigma.
    /// </summary>
    /// <param name="position"></param>
    /// <param name="pos0"></param>
    /// <param name="sigma"></param>
    /// <param name="scalarType"></param>
    /// <returns>
    /// A tensor of spike rates with shape (position.shape[0]).
    /// </returns>
    public static Tensor GaussianSpikeRates(Tensor position, Tensor center, Tensor sigma, ScalarType? scalarType = null)
    {
        using (var _ = NewDisposeScope())
        {
            var dist = position.unsqueeze(1) - center;
            var distSquared = dist.pow(2).sum(dim: -1);
            var sigmaSquared = sigma.pow(2);
            var distNormed = distSquared / (2.0 * sigmaSquared);
            var spikingRates = exp(-distNormed);
            if (scalarType != null) spikingRates = spikingRates.to_type(scalarType.Value);
            return spikingRates.MoveToOuterDisposeScope();
        }
    }

    /// <summary>
    /// Simulate spikes at a given position using place fields and a firing threshold.
    /// </summary>
    /// <param name="positionData"></param>
    /// <param name="placeFieldCenters"></param>
    /// <param name="placeFieldRadius"></param>
    /// <param name="firingThreshold"></param>
    /// <param name="seed"></param>
    /// <param name="scalarType"></param>
    /// <param name="noiseScale"></param>
    /// <returns>
    /// A tensor of spikes with shape (positionData.shape[0], placeFieldCenters.shape[0]).
    /// </returns>
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
            return spikes.MoveToOuterDisposeScope();
        }
    }
}