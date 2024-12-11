using static TorchSharp.torch;
using PointProcessDecoder.Core;
using PointProcessDecoder.Plot;
using PointProcessDecoder.Simulation;
using PointProcessDecoder.Core.Estimation;

namespace PointProcessDecoder.Test;

[TestClass]
public class TestEstimation
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
    private double bandwidth = 5;
    private int numDimensions = 1;
    private long evaluationSteps = 50;
    private double distanceThreshold = 1.5;
    private int heatmapPadding = 10;
    private string outputDirectory = "TestEstimation";

    [TestMethod]
    public void TestKernelCompression()
    {
        var kernelCompressionDirectory = Path.Combine(outputDirectory, "KernelCompression");

        var position1D = Simulate.Position(steps, cycles, yMin, yMax, scalarType);
        var position1DExpanded = concat([zeros_like(position1D), position1D], dim: 1);

        var placeFieldCenters = Simulate.PlaceFieldCenters(yMin, yMax, numNeurons, seed, scalarType);
        var placeFieldCenters2D = vstack([zeros_like(placeFieldCenters), placeFieldCenters]).T;

        var spikingData = Simulate.SpikesAtPosition(position1DExpanded, placeFieldCenters2D, placeFieldRadius, firingThreshold, seed);

        var positionKC = new KernelCompression(bandwidth, numDimensions, distanceThreshold, device);
        positionKC.Fit(position1D);
        var positionKCDensity = positionKC.Evaluate(
            new double[] { yMin }, 
            new double[] { yMax }, 
            new long[] { evaluationSteps }
        ).to_type(scalarType);

        var positionKC2D = tile(positionKCDensity, [heatmapPadding, 1]);

        Heatmap plotKernelDensityEstimate = new(
            0,
            heatmapPadding,
            yMin,
            yMax,      
            title: "PositionKernelCompression2D"
        );
        plotKernelDensityEstimate.OutputDirectory = Path.Combine(plotKernelDensityEstimate.OutputDirectory, kernelCompressionDirectory);
        plotKernelDensityEstimate.Show(positionKC2D);
        plotKernelDensityEstimate.Save(png: true);

        var neuronKDEs = new List<KernelCompression>();
        for (int i = 0; i < numNeurons; i++)
        {
            var neuronKDE = new KernelCompression(bandwidth, numDimensions, distanceThreshold, device);
            var neuronPosition1D = position1D[spikingData[TensorIndex.Colon, i]];
            neuronKDE.Fit(neuronPosition1D);
            neuronKDEs.Add(neuronKDE);

            var neuronDensity = neuronKDE.Evaluate(
                new double[] { yMin }, 
                new double[] { yMax }, 
                new long[] { evaluationSteps }
            ).to_type(scalarType);

            var neuronDensity2D = tile(neuronDensity, [heatmapPadding, 1]).T;

            var compressedPlaceFieldsDirectory = Path.Combine(kernelCompressionDirectory, "CompressedPlaceFields");

            Heatmap plotNeuronKernelDensityEstimate = new(
                yMin, 
                yMax, 
                0, 
                heatmapPadding, 
                title: $"Neuron {i} Place Field", 
                figureName: $"Neuron{i}PlaceField"
            );
            plotNeuronKernelDensityEstimate.OutputDirectory = Path.Combine(plotNeuronKernelDensityEstimate.OutputDirectory, compressedPlaceFieldsDirectory);
            plotNeuronKernelDensityEstimate.Show(neuronDensity2D);
            plotNeuronKernelDensityEstimate.Save(png: true);
        }
        Assert.IsTrue(true);
    }

    [TestMethod]
    public void TestKernelDensity()
    {
        var kernelDensityDirectory = Path.Combine(outputDirectory, "KernelDensity");

        var position1D = Simulate.Position(steps, cycles, yMin, yMax, scalarType);
        var position1DExpanded = concat([zeros_like(position1D), position1D], dim: 1);

        var placeFieldCenters = Simulate.PlaceFieldCenters(yMin, yMax, numNeurons, seed, scalarType);
        var placeFieldCenters2D = vstack([zeros_like(placeFieldCenters), placeFieldCenters]).T;

        var spikingData = Simulate.SpikesAtPosition(position1DExpanded, placeFieldCenters2D, placeFieldRadius, firingThreshold, seed);
        var positionKDE = new KernelDensity(bandwidth, numDimensions, device);
        positionKDE.Fit(position1D);
        var positionDensity = positionKDE.Evaluate(
            new double[] { yMin }, 
            new double[] { yMax }, 
            new long[] { evaluationSteps }
        ).to_type(scalarType);

        var positionDensity2DExtended = tile(positionDensity, [heatmapPadding, 1]);

        Heatmap plotKernelDensityEstimate = new(
            0,
            heatmapPadding, 
            yMin,
            yMax,    
            title: "PositionDensity2D"
        );
        plotKernelDensityEstimate.OutputDirectory = Path.Combine(plotKernelDensityEstimate.OutputDirectory, kernelDensityDirectory);
        plotKernelDensityEstimate.Show(positionDensity2DExtended);
        plotKernelDensityEstimate.Save(png: true);

        var neuronKDEs = new List<KernelDensity>();
        for (int i = 0; i < numNeurons; i++)
        {
            var neuronKDE = new KernelDensity(bandwidth, numDimensions);
            var neuronPosition1D = position1D[spikingData[TensorIndex.Colon, i]];
            neuronKDE.Fit(neuronPosition1D);
            neuronKDEs.Add(neuronKDE);

            var neuronDensity = neuronKDE.Evaluate(
                new double[] { yMin }, 
                new double[] { yMax }, 
                new long[] { evaluationSteps }
            ).to_type(scalarType);

            var neuronDensity2D = tile(neuronDensity, [heatmapPadding, 1]).T;

            var placeFieldsDirectory = Path.Combine(kernelDensityDirectory, "PlaceFields");

            Heatmap plotNeuronKernelDensityEstimate = new(
                yMin,
                yMax,
                0,
                heatmapPadding,    
                title: $"Neuron {i} Place Field", 
                figureName: $"Neuron{i}PlaceField"
            );
            plotNeuronKernelDensityEstimate.OutputDirectory = Path.Combine(plotNeuronKernelDensityEstimate.OutputDirectory, placeFieldsDirectory);
            plotNeuronKernelDensityEstimate.Show(neuronDensity2D);
            plotNeuronKernelDensityEstimate.Save(png: true);
        }
        Assert.IsTrue(true);
    }
}