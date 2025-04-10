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
    private static ScalarType _scalarType = ScalarType.Float32;
    public static ScalarType ScalarType
    {
        get => _scalarType;
        set => _scalarType = value;
    }

    private static Device _device = CPU;
    public static Device Device
    {
        get => _device;
        set => _device = value;
    }

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
    public static Tensor SinPosition(
        int steps, 
        int cycles, 
        double min, 
        double max, 
        ScalarType? scalarType = null, 
        Device? device = null
    )
    {
        scalarType ??= _scalarType;
        device ??= _device;
        using var _ = NewDisposeScope();
        return ((sin(linspace(0, 2.0 * Math.PI * cycles, steps * cycles)) + 1.0) * 0.5 * (max - min) + min)
            .unsqueeze(1)
            .to_type(scalarType.Value)
            .to(device)
            .MoveToOuterDisposeScope();
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
    public static Tensor SinPosition(
        int steps, 
        int cycles, 
        double xMin, 
        double xMax, 
        double yMin, 
        double yMax, 
        double scale = 1.0, 
        ScalarType? scalarType = null, 
        Device? device = null
    )
    {
        scalarType ??= _scalarType;
        device ??= _device;
        using var _ = NewDisposeScope();
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
        var positionData = vstack([x, y]).T;
        return positionData
            .to_type(scalarType.Value)
            .to(device)
            .MoveToOuterDisposeScope();
    }

    public static Tensor RandPosition(
        int count,
        double min, 
        double max,
        int? seed = null,
        ScalarType? scalarType = null, 
        Device? device = null
    )
    {
        using var _ = NewDisposeScope();
        if (seed != null) manual_seed(seed.Value);

        scalarType ??= _scalarType;
        device ??= _device;

        return (rand(count, 1) * max - min + min)
            .to_type(scalarType.Value)
            .to(device)
            .MoveToOuterDisposeScope();
    }

    public static Tensor RandPosition(
        int count,
        double xMin, 
        double xMax, 
        double yMin, 
        double yMax,
        int? seed = null,
        ScalarType? scalarType = null, 
        Device? device = null
    )
    {
        using var _ = NewDisposeScope();
        if (seed != null) manual_seed(seed.Value);

        scalarType ??= _scalarType;
        device ??= _device;

        return (rand(count, 2) * tensor(new double[] { xMax - xMin, yMax - yMin }) + tensor(new double[] { xMin, yMin }))
            .to_type(scalarType.Value)
            .to(device)
            .MoveToOuterDisposeScope();
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
    public static Tensor PlaceFieldCenters(
        double min, 
        double max, 
        int numNeurons,
        ScalarType? scalarType = null,
        Device? device = null
    )
    {
        scalarType ??= _scalarType;
        device ??= _device;
        using var _ = NewDisposeScope();
        var positions = linspace(min + 0.5 * (max - min) / numNeurons, max - 0.5 * (max - min) / numNeurons, numNeurons).unsqueeze(1);
        return positions
            .to_type(scalarType.Value)
            .to(device)
            .MoveToOuterDisposeScope();
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
    public static Tensor PlaceFieldCenters(
        double xMin, 
        double xMax, 
        double yMin, 
        double yMax, 
        int numNeurons, 
        int? seed = null, 
        ScalarType? scalarType = null,
        Device? device = null
    )
    {
        scalarType ??= _scalarType;
        device ??= _device;
        using var _ = NewDisposeScope();
        var generator = seed != null ? manual_seed(seed.Value) : null;

        var count = sqrt(numNeurons).to_type(ScalarType.Int32).item<int>();
        var xSpacing = linspace(xMin + 0.5 * (xMax - xMin) / count, xMax - 0.5 * (xMax - xMin) / count, count);
        var ySpacing = linspace(yMin + 0.5 * (yMax - yMin) / count, yMax - 0.5 * (yMax - yMin) / count, count);
        var positionGrid = meshgrid([ xSpacing, ySpacing ]);
        var centers = vstack([ positionGrid[0].flatten(), positionGrid[1].flatten() ]).T;
        var leftover = rand(numNeurons - count * count, 2, generator: generator) * tensor(new double[] { xMax - xMin, yMax - yMin }) + tensor(new double[] { xMin, yMin });
        var positions = vstack([ centers, leftover ]);
        return positions
            .to_type(scalarType.Value)
            .to(device)
            .MoveToOuterDisposeScope();
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
    public static Tensor GaussianSpikeRates(
        Tensor position, 
        Tensor center, 
        Tensor sigma, 
        ScalarType? scalarType = null,
        Device? device = null
    )
    {
        scalarType ??= _scalarType;
        device ??= _device;
        using var _ = NewDisposeScope();
        var dist = position.unsqueeze(1) - center;
        var distSquared = dist.pow(2).sum(dim: -1);
        var distNormed = distSquared / (2.0 * sigma.pow(2));
        var spikingRates = exp(-distNormed);
        return spikingRates
            .to_type(scalarType.Value)
            .to(device)
            .MoveToOuterDisposeScope();
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
    public static Tensor SpikesAtPosition(
        Tensor positionData, 
        Tensor placeFieldCenters, 
        Tensor placeFieldRadius, 
        Tensor firingThreshold, 
        int? seed = null, 
        ScalarType? scalarType = ScalarType.Int32,
        Device? device = null,
        double? noiseScale = null)
    {
        scalarType ??= _scalarType;
        device ??= _device;
        using var _ = NewDisposeScope();
        var generator = seed != null ? manual_seed(seed.Value) : null;
        var spikingRates = GaussianSpikeRates(positionData, placeFieldCenters, placeFieldRadius);
        var noise = rand(spikingRates.shape, generator: generator);
        noise *= noiseScale ?? 1.0;
        var spikeThreshold = firingThreshold + noise;
        var spikes = spikeThreshold < spikingRates;
        return spikes
            .to_type(scalarType.Value)
            .to(device)
            .MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Simulate marks at a given position using spikes and mark dimensions.
    /// </summary>
    /// <param name="positionData"></param>
    /// <param name="spikes"></param>
    /// <param name="markDimensions"></param>
    /// <param name="markChannels"></param>
    /// <param name="seed"></param>
    /// <param name="scalarType"></param>
    /// <param name="device"></param>
    /// <param name="noiseScale"></param>
    /// <returns></returns>
    public static Tensor MarksAtPosition(
        Tensor positionData,
        Tensor spikes,
        int markDimensions,
        int markChannels,
        int? seed = null,
        ScalarType? scalarType = null,
        Device? device = null,
        double spikeScale = 5,
        double noiseScale = 0.5
    )
    {
        scalarType ??= _scalarType;
        device ??= _device;
        using var _ = NewDisposeScope();
        var generator = seed != null ? manual_seed(seed.Value) : null;
        var marks = ones([positionData.shape[0], markDimensions, markChannels], device: device, dtype: scalarType.Value) * double.NaN;
        var nUnits = spikes.shape[1];
        var spikeIndices = spikes.nonzero();
        var unitMarks = rand([nUnits, markDimensions], device: device, dtype: scalarType.Value, generator: generator) * spikeScale;
        var neuronsPerChannel = (int)Math.Ceiling((double)nUnits / markChannels);

        // simulate marks for each channel
        for (int i = 0; i < markChannels; i++)
        {
            // we expect there to be more units than channels, and each unit should have a unique mark with some noise
            for (int j = 0; j < neuronsPerChannel; j++)
            {
                if (j + i * neuronsPerChannel >= nUnits) break;
                var unitIndex = j + i * neuronsPerChannel;
                var unitSpikes = spikeIndices[spikeIndices[TensorIndex.Colon, 1] == unitIndex];
                var unitMark = unitMarks[unitIndex];
                var unitMarkExpanded = unitMark.unsqueeze(0).expand(unitSpikes.shape[0], markDimensions);
                // add gaussian noise to the mark
                unitMarkExpanded += randn(unitMarkExpanded.shape, generator: generator) * noiseScale;
                marks[TensorIndex.Tensor(unitSpikes[TensorIndex.Colon, 0]), TensorIndex.Colon, i] = unitMarkExpanded;
            }
        }
        return marks.MoveToOuterDisposeScope();
    }
}