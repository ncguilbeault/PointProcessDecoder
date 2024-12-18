using static TorchSharp.torch;
using PointProcessDecoder.Core;
using PointProcessDecoder.Core.Estimation;
using PointProcessDecoder.Core.Encoder;
using PointProcessDecoder.Core.StateSpace;

namespace PointProcessDecoder.Test.Common;

public static class TestEncoder
{
    public static void KernelDensity(
        double[]? bandwidth = null,
        int numDimensions = 1,
        long[]? evaluationSteps = null,
        int steps = 200,
        int cycles = 10,
        double[]? min = null,
        double[]? max = null,
        int numNeurons = 40,
        double placeFieldRadius = 8.0,
        double firingThreshold = 0.2,
        string outputDirectory = "TestEncoder",
        string modelDirectory = "KernelDensity",
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

        outputDirectory = string.IsNullOrEmpty(modelDirectory) ? outputDirectory : Path.Combine(outputDirectory, modelDirectory);

        var stateSpace = new DiscreteUniformStateSpace(
            numDimensions,
            min,
            max,
            evaluationSteps,
            device: device,
            scalarType: scalarType
        );

        var sortedSpikeEncoder = new SortedSpikeEncoder(
            EstimationMethod.KernelDensity, 
            bandwidth,
            numNeurons,
            stateSpace,
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

    public static void KernelCompression(
        double[]? bandwidth = null,
        int numDimensions = 1,
        long[]? evaluationSteps = null,
        int steps = 200,
        int cycles = 10,
        double[]? min = null,
        double[]? max = null,
        int numNeurons = 40,
        double placeFieldRadius = 8.0,
        double firingThreshold = 0.2,
        double distanceThreshold = 1.5,
        string outputDirectory = "TestEncoder",
        string modelDirectory = "KernelCompression",
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

        outputDirectory = string.IsNullOrEmpty(modelDirectory) ? outputDirectory : Path.Combine(outputDirectory, modelDirectory);

        var stateSpace = new DiscreteUniformStateSpace(
            numDimensions,
            min,
            max,
            evaluationSteps,
            device: device,
            scalarType: scalarType
        );

        var sortedSpikeEncoder = new SortedSpikeEncoder(
            EstimationMethod.KernelCompression, 
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
}