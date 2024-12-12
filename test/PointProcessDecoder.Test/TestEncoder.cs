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
    private int heatmapPadding = 10;
    private int seed = 0;
    private ScalarType scalarType = ScalarType.Float32;
    private Device device = CPU;
    private string outputDirectory = "TestEncoder";

    private (Tensor, Tensor) InitializeSimulation1D(
        int steps = 200,
        int cycles = 10,
        double min = 0.0,
        double max = 100.0,
        int numNeurons = 40,
        double placeFieldRadius = 8.0,
        double firingThreshold = 0.2
    )
    {
        var position1D = Simulate.Position(steps, cycles, min, max, scalarType);
        var position1DExpanded = concat([zeros_like(position1D), position1D], dim: 1);

        var placeFieldCenters = Simulate.PlaceFieldCenters(min, max, numNeurons, seed, scalarType);
        var placeFieldCenters2D = concat([zeros_like(placeFieldCenters), placeFieldCenters], dim: 1);

        var spikingData = Simulate.SpikesAtPosition(position1DExpanded, placeFieldCenters2D, placeFieldRadius, firingThreshold, seed);

        return (position1D, spikingData);
    }

    private (Tensor, Tensor) InitializeSimulation2D(
        int steps = 200,
        int cycles = 10,
        double xMin = 0.0,
        double xMax = 100.0,
        double yMin = 0.0,
        double yMax = 100.0,
        int numNeurons = 40,
        double placeFieldRadius = 8.0,
        double firingThreshold = 0.2
    )
    {
        var position2D = Simulate.Position(steps, cycles, xMin, xMax, yMin, yMax, scalarType: scalarType);
        var placeFieldCenters = Simulate.PlaceFieldCenters(xMin, yMax, yMin, yMax, numNeurons, seed, scalarType);
        var spikingData = Simulate.SpikesAtPosition(position2D, placeFieldCenters, placeFieldRadius, firingThreshold, seed);

        return (position2D, spikingData);
    }

    private void RunSortedSpikeEncoder1D(
        IEncoder encoder, 
        Tensor observations, 
        Tensor spikes,
        string encoderDirectory,
        long evaluationSteps,
        double[] densityScatterPlotRange,
        double[] densityHeatmapRange
    )
    {
        encoder.Encode(observations, spikes);
        var densities = encoder.Evaluate();

        for (int i = 0; i < densities.Count(); i++)
        {
            var density = densities.ElementAt(i);
            var density1DExpanded = vstack([arange(evaluationSteps), density]).T;

            var directoryScatterPlot1D = Path.Combine(encoderDirectory, "ScatterPlot1D");

            ScatterPlot plotDensity1D = new ScatterPlot(
                densityScatterPlotRange[0],
                densityScatterPlotRange[1],
                densityScatterPlotRange[2],
                densityScatterPlotRange[3],
                title: $"Density1D{i}"
            );
            plotDensity1D.OutputDirectory = Path.Combine(plotDensity1D.OutputDirectory, directoryScatterPlot1D);
            plotDensity1D.Show<float>(density1DExpanded);
            plotDensity1D.Save(png: true);

            var density2D = tile(density, [heatmapPadding, 1]);

            var directoryHeatmap2D = Path.Combine(encoderDirectory, "Heatmap2D");

            Heatmap plotDensity2D = new(
                densityHeatmapRange[0],
                densityHeatmapRange[1],
                densityHeatmapRange[2],
                densityHeatmapRange[3],
                title: $"SortedUnitDensity2D{i}"
            );
            plotDensity2D.OutputDirectory = Path.Combine(plotDensity2D.OutputDirectory, directoryHeatmap2D);
            plotDensity2D.Show<float>(density2D);
            plotDensity2D.Save(png: true);
        }
    }

    private void RunSortedSpikeEncoder2D(
        IEncoder encoder, 
        Tensor observations, 
        Tensor spikes,
        string encoderDirectory,
        double[] densityHeatmapRange
    )
    {
        encoder.Encode(observations, spikes);
        var densities = encoder.Evaluate();

        for (int i = 0; i < densities.Count(); i++)
        {
            var density = densities.ElementAt(i);

            var directoryHeatmap2D = Path.Combine(encoderDirectory, "Heatmap2D");

            Heatmap plotDensity2D = new(
                densityHeatmapRange[0],
                densityHeatmapRange[1],
                densityHeatmapRange[2],
                densityHeatmapRange[3],
                title: $"SortedUnitDensity2D{i}"
            );

            plotDensity2D.OutputDirectory = Path.Combine(plotDensity2D.OutputDirectory, directoryHeatmap2D);
            plotDensity2D.Show<float>(density);
            plotDensity2D.Save(png: true);
        }
    }

    [TestMethod]
    public void TestSortedSpikeEncoderKernelDensity()
    {
        double[] bandwidth = [5.0];
        int numDimensions = 1;
        long evaluationSteps = 50;
        int steps = 200;
        int cycles = 10;
        double min = 0.0;
        double max = 100.0;
        int numNeurons = 40;
        double placeFieldRadius = 8.0;
        double firingThreshold = 0.2;

        var sortedSpikeEncoderDirectory = Path.Combine(outputDirectory, "SortedSpikeEncoderKernelDensity");
        var (position1D, spikingData) = InitializeSimulation1D(
            steps,
            cycles,
            min,
            max,
            numNeurons,
            placeFieldRadius,
            firingThreshold
        );

        var sortedSpikeEncoder = new SortedSpikeEncoder(
            EstimationMethod.KernelDensity, 
            bandwidth, 
            numDimensions,
            numNeurons,
            [min], 
            [max], 
            [evaluationSteps],
            device: device,
            scalarType: scalarType
        );

        RunSortedSpikeEncoder1D(
            sortedSpikeEncoder, 
            position1D, 
            spikingData, 
            sortedSpikeEncoderDirectory,
            evaluationSteps,
            [ 0, evaluationSteps, 0, 1 ],
            [ 0, heatmapPadding, min, max ]
        );

        Assert.IsTrue(true);
    }

    [TestMethod]
    public void TestSortedSpikeEncoderKernelCompression()
    {
        double[] bandwidth = [5.0];
        int numDimensions = 1;
        long evaluationSteps = 50;
        int steps = 200;
        int cycles = 10;
        double min = 0.0;
        double max = 100.0;
        int numNeurons = 40;
        double placeFieldRadius = 8.0;
        double firingThreshold = 0.2;
        double distanceThreshold = 1.5;

        var sortedSpikeEncoderDirectory = Path.Combine(outputDirectory, "SortedSpikeEncoderKernelCompression");
        var (position1D, spikingData) = InitializeSimulation1D(
            steps,
            cycles,
            min,
            max,
            numNeurons,
            placeFieldRadius,
            firingThreshold
        );

        var sortedSpikeEncoder = new SortedSpikeEncoder(
            EstimationMethod.KernelCompression, 
            bandwidth, 
            numDimensions,
            numNeurons,
            [min], 
            [max], 
            [evaluationSteps],
            distanceThreshold: distanceThreshold,
            device: device
        );

        RunSortedSpikeEncoder1D(
            sortedSpikeEncoder, 
            position1D, 
            spikingData, 
            sortedSpikeEncoderDirectory,
            evaluationSteps,
            [ 0, evaluationSteps, 0, 1 ],
            [ 0, heatmapPadding, min, max ]
        );

        Assert.IsTrue(true);
    }

    [TestMethod]
    public void TestSortedSpikeEncoderKernelCompression2D()
    {
        double[] bandwidth = [5.0, 5.0];
        int numDimensions = 2;
        long[] evaluationSteps = [50, 50];
        var steps = 200;
        var cycles = 10;
        var xMin = 0.0;
        var xMax = 100.0;
        var yMin = 0.0;
        var yMax = 100.0;

        int numNeurons = 40;
        double placeFieldRadius = 8.0;
        double firingThreshold = 0.2;
        double distanceThreshold = 1.5;

        var sortedSpikeEncoderDirectory = Path.Combine(outputDirectory, "SortedSpikeEncoderKernelCompression2D");
        var (position2D, spikingData) = InitializeSimulation2D(
            steps,
            cycles,
            xMin,
            xMax,
            yMin,
            yMax,
            numNeurons,
            placeFieldRadius,
            firingThreshold
        );

        var sortedSpikeEncoder = new SortedSpikeEncoder(
            EstimationMethod.KernelCompression, 
            bandwidth, 
            numDimensions,
            numNeurons,
            [xMin, yMin], 
            [xMax, yMax], 
            evaluationSteps,
            distanceThreshold: distanceThreshold,
            device: device
        );

        RunSortedSpikeEncoder2D(
            sortedSpikeEncoder, 
            position2D, 
            spikingData, 
            sortedSpikeEncoderDirectory,
            [ xMin, xMax, yMin, yMax ]
        );

        Assert.IsTrue(true);
    }

    [TestMethod]
    public void TestSortedSpikeEncoderKernelDensity2D()
    {
        double[] bandwidth = [5.0, 5.0];
        int numDimensions = 2;
        long[] evaluationSteps = [50, 50];
        var steps = 200;
        var cycles = 10;
        var xMin = 0.0;
        var xMax = 100.0;
        var yMin = 0.0;
        var yMax = 100.0;

        int numNeurons = 40;
        double placeFieldRadius = 8.0;
        double firingThreshold = 0.2;

        var sortedSpikeEncoderDirectory = Path.Combine(outputDirectory, "SortedSpikeEncoderKernelDensity2D");
        var (position2D, spikingData) = InitializeSimulation2D(
            steps,
            cycles,
            xMin,
            xMax,
            yMin,
            yMax,
            numNeurons,
            placeFieldRadius,
            firingThreshold
        );

        var sortedSpikeEncoder = new SortedSpikeEncoder(
            EstimationMethod.KernelDensity, 
            bandwidth, 
            numDimensions,
            numNeurons,
            [xMin, yMin], 
            [xMax, yMax], 
            evaluationSteps,
            device: device
        );

        RunSortedSpikeEncoder2D(
            sortedSpikeEncoder, 
            position2D, 
            spikingData, 
            sortedSpikeEncoderDirectory,
            [ xMin, xMax, yMin, yMax ]
        );

        Assert.IsTrue(true);
    }
}