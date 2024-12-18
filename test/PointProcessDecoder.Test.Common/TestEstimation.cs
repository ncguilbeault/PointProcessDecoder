using static TorchSharp.torch;
using PointProcessDecoder.Plot;
using PointProcessDecoder.Simulation;
using PointProcessDecoder.Core.Estimation;

namespace PointProcessDecoder.Test.Common;

public static class TestEstimation
{
    public static void KernelCompression(
        int seed = 0,
        int steps = 200,
        int cycles = 10,
        double yMin = 0.0,
        double yMax = 100.0,
        ScalarType scalarType = ScalarType.Float32,
        Device? device = null,
        int numNeurons = 40,
        double placeFieldRadius = 8.0,
        double firingThreshold = 0.2,
        double bandwidth = 5,
        int numDimensions = 1,
        long evaluationSteps = 50,
        double distanceThreshold = 1.5,
        int heatmapPadding = 10,
        string outputDirectory = "TestEstimation"
    )
    {
        device ??= CPU;

        var kernelCompressionDirectory = Path.Combine(outputDirectory, "KernelCompression1D");

        var position1D = Simulate.Position(steps, cycles, yMin, yMax, scalarType);
        var position1DExpanded = concat([zeros_like(position1D), position1D], dim: 1);

        var placeFieldCenters = Simulate.PlaceFieldCenters(yMin, yMax, numNeurons, seed, scalarType);
        var placeFieldCenters2D = concat([zeros_like(placeFieldCenters), placeFieldCenters], dim: 1);

        var spikingData = Simulate.SpikesAtPosition(position1DExpanded, placeFieldCenters2D, placeFieldRadius, firingThreshold, seed);

        var positionKC = new KernelCompression(bandwidth, numDimensions, distanceThreshold, device, scalarType);
        positionKC.Fit(position1D.to_type(ScalarType.Float32));
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
        plotKernelDensityEstimate.Show<float>(positionKC2D);
        plotKernelDensityEstimate.Save(png: true);

        var neuronKDEs = new List<KernelCompression>();
        for (int i = 0; i < numNeurons; i++)
        {
            var neuronKDE = new KernelCompression(bandwidth, numDimensions, distanceThreshold, device, scalarType);
            var neuronPosition1D = position1D[spikingData[TensorIndex.Colon, i]];
            neuronKDE.Fit(neuronPosition1D.to_type(ScalarType.Float64));
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
            plotNeuronKernelDensityEstimate.Show<float>(neuronDensity2D);
            plotNeuronKernelDensityEstimate.Save(png: true);
        }
    }

    public static void KernelDensity(
        int seed = 0,
        int steps = 200,
        int cycles = 10,
        double yMin = 0.0,
        double yMax = 100.0,
        ScalarType scalarType = ScalarType.Float32,
        Device? device = null,
        int numNeurons = 40,
        double placeFieldRadius = 8.0,
        double firingThreshold = 0.2,
        double bandwidth = 5,
        int numDimensions = 1,
        long evaluationSteps = 50,
        int heatmapPadding = 10,
        string outputDirectory = "TestEstimation"
    )
    {
        device ??= CPU;

        var kernelDensityDirectory = Path.Combine(outputDirectory, "KernelDensity");

        var position1D = Simulate.Position(steps, cycles, yMin, yMax, scalarType);
        var position1DExpanded = concat([zeros_like(position1D), position1D], dim: 1);

        var placeFieldCenters = Simulate.PlaceFieldCenters(yMin, yMax, numNeurons, seed, scalarType);
        var placeFieldCenters2D = concat([zeros_like(placeFieldCenters), placeFieldCenters], dim: 1);

        var spikingData = Simulate.SpikesAtPosition(position1DExpanded, placeFieldCenters2D, placeFieldRadius, firingThreshold, seed);
        var positionKDE = new KernelDensity(bandwidth, numDimensions, device);
        positionKDE.Fit(position1D.to_type(ScalarType.Float64));
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
        plotKernelDensityEstimate.Show<float>(positionDensity2DExtended);
        plotKernelDensityEstimate.Save(png: true);

        var neuronKDEs = new List<KernelDensity>();
        for (int i = 0; i < numNeurons; i++)
        {
            var neuronKDE = new KernelDensity(bandwidth, numDimensions);
            var neuronPosition1D = position1D[spikingData[TensorIndex.Colon, i]];
            neuronKDE.Fit(neuronPosition1D.to_type(ScalarType.Float64));
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
            plotNeuronKernelDensityEstimate.Show<float>(neuronDensity2D);
            plotNeuronKernelDensityEstimate.Save(png: true);
        }
    }
}