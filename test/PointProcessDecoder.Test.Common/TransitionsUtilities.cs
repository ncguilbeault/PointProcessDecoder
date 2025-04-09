using static TorchSharp.torch;

using PointProcessDecoder.Plot;
using PointProcessDecoder.Core.Transitions;
using PointProcessDecoder.Core.StateSpace;

namespace PointProcessDecoder.Test.Common;

public static class TransitionsUtilities
{
    public static void TestUniform(
        int dimensions = 1,
        long[]? steps = null,
        double[]? min = null,
        double[]? max = null,
        string outputDirectory = "TestTransitions",
        string figureName = "UniformTransitions",
        ScalarType scalarType = ScalarType.Float32,
        Device? device = null
    )
    {
        steps ??= [100];
        min ??= [0];
        max ??= [100];
        device ??= CPU;

        var stateSpace = new DiscreteUniform(
            dimensions,
            min,
            max,
            steps,
            device: device,
            scalarType: scalarType
        );

        var uniformTransitions = new Uniform(
            stateSpace,
            device: device,
            scalarType: scalarType
        );

        var xIdx = 0;
        var yIdx = dimensions == 2 ? 1 : 0;

        Heatmap transitionsPlot = new(min[xIdx], max[xIdx], min[yIdx], max[yIdx], title: figureName);
        transitionsPlot.OutputDirectory = Path.Combine(transitionsPlot.OutputDirectory, outputDirectory);
        transitionsPlot.Show(uniformTransitions.Transitions);
        transitionsPlot.Save(png: true);
    }

    public static void TestRandomWalk(
        int dimensions = 1,
        long[]? steps = null,
        double[]? min = null,
        double[]? max = null,
        double? sigma = null,
        string outputDirectory = "TestTransitions",
        string figureName = "RandomWalkTransitions",
        ScalarType scalarType = ScalarType.Float32,
        Device? device = null
    )
    {
        steps ??= [100];
        min ??= [0];
        max ??= [100];
        device ??= CPU;

        var stateSpace = new DiscreteUniform(
            dimensions,
            min,
            max,
            steps,
            device: device,
            scalarType: scalarType
        );

        var randomWalkTransitions = new RandomWalk(
            stateSpace,
            sigma,
            device: device,
            scalarType: scalarType
        );

        var xIdx = 0;
        var yIdx = dimensions == 2 ? 1 : 0;

        Heatmap transitionsPlot = new(min[xIdx], max[xIdx], min[yIdx], max[yIdx], title: figureName);
        transitionsPlot.OutputDirectory = Path.Combine(transitionsPlot.OutputDirectory, outputDirectory);
        transitionsPlot.Show(randomWalkTransitions.Transitions);
        transitionsPlot.Save(png: true);
    }
}