using static TorchSharp.torch;
using PointProcessDecoder.Plot;
using PointProcessDecoder.Core.Estimation;
using PointProcessDecoder.Core.Encoder;
using PointProcessDecoder.Core.StateSpace;
using PointProcessDecoder.Simulation;

namespace PointProcessDecoder.Test.Common;

public static class EncoderUtilities
{
    public static void SortedSpikeEncoder(
        EstimationMethod estimationMethod = EstimationMethod.KernelDensity,
        double[]? bandwidth = null,
        int numDimensions = 1,
        long[]? evaluationSteps = null,
        int steps = 200,
        int cycles = 10,
        double[]? min = null,
        double[]? max = null,
        double? distanceThreshold = null,
        int numNeurons = 40,
        double placeFieldRadius = 8.0,
        double firingThreshold = 0.2,
        string outputDirectory = "TestEncoder",
        string modelDirectory = "SortedSpikeEncoder",
        ScalarType scalarType = ScalarType.Float32,
        Device? device = null,
        int heatmapPadding = 10
    )
    {
        bandwidth ??= [5.0];
        evaluationSteps ??= [50];
        min ??= [0.0];
        max ??= [100.0];
        device ??= CPU;

        outputDirectory = Path.Combine(outputDirectory, modelDirectory, estimationMethod.ToString());

        var stateSpace = new DiscreteUniformStateSpace(
            numDimensions,
            min,
            max,
            evaluationSteps,
            device: device,
            scalarType: scalarType
        );

        var sortedSpikeEncoder = new SortedSpikeEncoder(
            estimationMethod, 
            bandwidth,
            numNeurons,
            stateSpace,
            distanceThreshold: distanceThreshold,
            device: device,
            scalarType: scalarType
        );

        if (numDimensions == 1)
        {
            var (position1D, spikingData) = Utilities.InitializeSimulation1D(
                steps,
                cycles,
                min[0],
                max[0],
                numNeurons,
                placeFieldRadius,
                firingThreshold
            );

            Utilities.RunSortedSpikeEncoder1D(
                sortedSpikeEncoder, 
                position1D, 
                spikingData, 
                outputDirectory,
                evaluationSteps[0],
                [ 0, evaluationSteps[0], 0, 1 ],
                [ 0, heatmapPadding, min[0], max[0] ],
                heatmapPadding
            );
        }
        else if (numDimensions == 2)
        {
            var (position2D, spikingData) = Utilities.InitializeSimulation2D(
                steps,
                cycles,
                min[0],
                max[0],
                min[1],
                max[1],
                numNeurons,
                placeFieldRadius,
                firingThreshold
            );

            Utilities.RunSortedSpikeEncoder2D(
                sortedSpikeEncoder, 
                position2D, 
                spikingData, 
                outputDirectory,
                [ min[0], max[0], min[1], max[1] ]
            );
        }
    }

    public static void ClusterlessMarkEncoder(
        EstimationMethod estimationMethod = EstimationMethod.KernelDensity,
        double[]? observationBandwidth = null,
        int numDimensions = 1,
        long[]? evaluationSteps = null,
        int steps = 200,
        int cycles = 10,
        double[]? min = null,
        double[]? max = null,
        double? distanceThreshold = null,
        int markDimensions = 4,
        int markChannels = 8,
        double[]? markBandwidth = null,
        double placeFieldRadius = 8.0,
        double firingThreshold = 0.2,
        int numNeurons = 40,
        double spikeScale = 5.0,
        double noiseScale = 0.5,
        string outputDirectory = "TestEncoder",
        string modelDirectory = "ClusterlessMarkEncoder",
        ScalarType scalarType = ScalarType.Float32,
        Device? device = null,
        int heatmapPadding = 10
    )
    {
        observationBandwidth ??= [5.0];
        evaluationSteps ??= [50];
        markBandwidth ??= [1.0, 1.0, 1.0, 1.0];
        min ??= [0.0];
        max ??= [100.0];
        device ??= CPU;

        outputDirectory = Path.Combine(outputDirectory, modelDirectory, estimationMethod.ToString());

        var stateSpace = new DiscreteUniformStateSpace(
            numDimensions,
            min,
            max,
            evaluationSteps,
            device: device,
            scalarType: scalarType
        );

        var encoder = new ClusterlessMarkEncoder(
            estimationMethod, 
            observationBandwidth,
            markDimensions,
            markChannels,
            markBandwidth,
            stateSpace,
            distanceThreshold: distanceThreshold,
            device: device,
            scalarType: scalarType
        );

        Tensor position = empty(0);
        Tensor spikingData = empty(0);
        double[] heatmapRange = new double[4];
        bool is2D = numDimensions == 2;

        if (numDimensions == 1)
        {
            (position, spikingData) = Utilities.InitializeSimulation1D(
                steps,
                cycles,
                min[0],
                max[0],
                numNeurons,
                placeFieldRadius,
                firingThreshold
            );
            heatmapRange = [0, heatmapPadding, min[0], max[0]];
        }

        else if (numDimensions == 2)
        {
            (position, spikingData) = Utilities.InitializeSimulation2D(
                steps,
                cycles,
                min[0],
                max[0],
                min[1],
                max[1],
                numNeurons,
                placeFieldRadius,
                firingThreshold
            );
            heatmapRange = [min[0], max[0], min[1], max[1]];
        }

        var marks = Simulate.MarksAtPosition(
            position,
            spikingData,
            markDimensions,
            markChannels,
            spikeScale: spikeScale,
            noiseScale: noiseScale,
            scalarType: scalarType,
            device: device
        );

        encoder.Encode(position, marks);
        var channelDensities = exp(encoder.Evaluate().ElementAt(0));

        for (int i = 0; i < channelDensities.shape[0]; i++)
        {
            var density = channelDensities[i]
                .reshape(evaluationSteps);

            if (!is2D)
            {
                density = tile(density, [heatmapPadding, 1]);
            }

            Heatmap plotDensity2D = new(
                heatmapRange[0],
                heatmapRange[1],
                heatmapRange[2],
                heatmapRange[3],
                title: $"Heatmap2D_{i}"
            );

            plotDensity2D.OutputDirectory = Path.Combine(plotDensity2D.OutputDirectory, outputDirectory);
            plotDensity2D.Show<float>(density);
            plotDensity2D.Save(png: true);
        }
    }
}