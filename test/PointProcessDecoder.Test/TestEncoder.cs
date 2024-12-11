using static TorchSharp.torch;
using PointProcessDecoder.Core;
using PointProcessDecoder.Plot;
using PointProcessDecoder.Simulation;
using PointProcessDecoder.Core.Estimation;
using PointProcessDecoder.Core.Encoder;

namespace PointProcessDecoder.Test;

[TestClass]
public class TestEncoder
{
    private int seed = 0;
    private int steps = 200;
    private int cycles = 10;
    private double yMin = 0.0;
    private double yMax = 100.0;
    private ScalarType scalarType = ScalarType.Float64;
    private Device device = CPU;
    private int numNeurons = 40;
    private double placeFieldRadius = 8.0;
    private double firingThreshold = 0.2;
    private double bandwidth = 5.0;
    private int numDimensions = 1;
    private long evaluationSteps = 50;
    private int heatmapPadding = 10;
    private double distanceThreshold = 1.5;
    private string outputDirectory = "TestEncoder";

    private (Tensor, Tensor) InitializeSimulation1D()
    {
        var position1D = Simulate.Position(steps, cycles, yMin, yMax, scalarType);
        var position1DExpanded = concat([zeros_like(position1D), position1D], dim: 1);

        var placeFieldCenters = Simulate.PlaceFieldCenters(yMin, yMax, numNeurons, seed, scalarType);
        var placeFieldCenters2D = vstack([zeros_like(placeFieldCenters), placeFieldCenters]).T;

        var spikingData = Simulate.SpikesAtPosition(position1DExpanded, placeFieldCenters2D, placeFieldRadius, firingThreshold, seed);

        return (position1D, spikingData);
    }

    private void RunSortedUnitEncoder(EncoderModel encoder, Tensor observations, Tensor spikes, string encoderDirectory)
    {
        encoder.Encode(observations, spikes);
        var densities = encoder.Evaluate();

        for (int i = 0; i < densities.Count(); i++)
        {
            var density = densities.ElementAt(i);
            var density1DExpanded = vstack([arange(evaluationSteps), density]).T;

            var directoryScatterPlot1D = Path.Combine(encoderDirectory, "ScatterPlot1D");

            ScatterPlot plotDensity1D = new ScatterPlot(
                0, 
                evaluationSteps,
                0,
                1,
                title: $"Density1D{i}"
            );
            plotDensity1D.OutputDirectory = Path.Combine(plotDensity1D.OutputDirectory, directoryScatterPlot1D);
            plotDensity1D.Show(density1DExpanded);
            plotDensity1D.Save(png: true);

            var density2D = tile(density, [heatmapPadding, 1]);

            var directoryHeatmap2D = Path.Combine(encoderDirectory, "Heatmap2D");

            Heatmap plotDensity2D = new(
                0,
                heatmapPadding,
                yMin,
                yMax,
                title: $"SortedUnitDensity2D{i}"
            );
            plotDensity2D.OutputDirectory = Path.Combine(plotDensity2D.OutputDirectory, directoryHeatmap2D);
            plotDensity2D.Show(density2D);
            plotDensity2D.Save(png: true);
        }
    }

    [TestMethod]
    public void TestSortedUnitEncoderKernelDensity()
    {
        var sortedUnitEncoderDirectory = Path.Combine(outputDirectory, "SortedUnitEncoderKernelDensity");
        var (position1D, spikingData) = InitializeSimulation1D();

        var sortedUnitEncoder = new SortedUnitEncoder(
            EstimationMethod.KernelDensity, 
            [bandwidth], 
            numDimensions,
            numNeurons,
            [yMin], 
            [yMax], 
            [evaluationSteps],
            device: device
        );

        RunSortedUnitEncoder(sortedUnitEncoder, position1D, spikingData, sortedUnitEncoderDirectory);

        Assert.IsTrue(true);
    }

[TestMethod]
    public void TestSortedUnitEncoderKernelCompression()
    {
        var sortedUnitEncoderDirectory = Path.Combine(outputDirectory, "SortedUnitEncoderKernelCompression");
        var (position1D, spikingData) = InitializeSimulation1D();

        var sortedUnitEncoder = new SortedUnitEncoder(
            EstimationMethod.KernelCompression, 
            [bandwidth], 
            numDimensions,
            numNeurons,
            [yMin], 
            [yMax], 
            [evaluationSteps],
            distanceThreshold: distanceThreshold,
            device: device
        );

        RunSortedUnitEncoder(sortedUnitEncoder, position1D, spikingData, sortedUnitEncoderDirectory);

        Assert.IsTrue(true);
    }
}