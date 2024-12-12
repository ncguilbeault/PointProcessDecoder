using static TorchSharp.torch;
using PointProcessDecoder.Simulation;

namespace PointProcessDecoder.Test;

public static class Utilities
{
    
    public static (Tensor, Tensor) InitializeSimulation1D(
        int steps = 200,
        int cycles = 10,
        double min = 0.0,
        double max = 100.0,
        int numNeurons = 40,
        double placeFieldRadius = 8.0,
        double firingThreshold = 0.2,
        ScalarType scalarType = ScalarType.Float32,
        int seed = 0
    )
    {
        var position1D = Simulate.Position(steps, cycles, min, max, scalarType);
        var position1DExpanded = concat([zeros_like(position1D), position1D], dim: 1);

        var placeFieldCenters = Simulate.PlaceFieldCenters(min, max, numNeurons, seed, scalarType);
        var placeFieldCenters2D = concat([zeros_like(placeFieldCenters), placeFieldCenters], dim: 1);

        var spikingData = Simulate.SpikesAtPosition(position1DExpanded, placeFieldCenters2D, placeFieldRadius, firingThreshold, seed);

        return (position1D, spikingData);
    }

    public static (Tensor, Tensor) InitializeSimulation2D(
        int steps = 200,
        int cycles = 10,
        double xMin = 0.0,
        double xMax = 100.0,
        double yMin = 0.0,
        double yMax = 100.0,
        int numNeurons = 40,
        double placeFieldRadius = 8.0,
        double firingThreshold = 0.2,
        double scale = 1.0,
        ScalarType scalarType = ScalarType.Float32,
        int seed = 0
    )
    {
        var position2D = Simulate.Position(steps, cycles, xMin, xMax, yMin, yMax, scale: scale, scalarType: scalarType);
        var placeFieldCenters = Simulate.PlaceFieldCenters(xMin, yMax, yMin, yMax, numNeurons, seed, scalarType);
        var spikingData = Simulate.SpikesAtPosition(position2D, placeFieldCenters, placeFieldRadius, firingThreshold, seed);

        return (position2D, spikingData);
    }

}
