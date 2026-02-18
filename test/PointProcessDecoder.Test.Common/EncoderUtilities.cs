using static TorchSharp.torch;
using PointProcessDecoder.Plot;
using PointProcessDecoder.Core;
using PointProcessDecoder.Core.Estimation;
using PointProcessDecoder.Core.Encoder;
using PointProcessDecoder.Core.StateSpace;
using PointProcessDecoder.Simulation;

namespace PointProcessDecoder.Test.Common;

public static class EncoderUtilities
{
    public static void SortedSpikeEncoder(
        EstimationMethod estimationMethod = EstimationMethod.KernelDensity,
        double[]? covariateBandwidth = null,
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
        covariateBandwidth ??= [5.0];
        evaluationSteps ??= [50];
        min ??= [0.0];
        max ??= [100.0];
        device ??= CPU;

        outputDirectory = Path.Combine(outputDirectory, modelDirectory, estimationMethod.ToString());

        var stateSpace = new DiscreteUniform(
            numDimensions,
            min,
            max,
            evaluationSteps,
            device: device,
            scalarType: scalarType
        );

        var sortedSpikeEncoder = new SortedSpikes(
            estimationMethod, 
            covariateBandwidth,
            numNeurons,
            stateSpace,
            distanceThreshold: distanceThreshold,
            device: device,
            scalarType: scalarType
        );

        if (numDimensions == 1)
        {
            var (position1D, spikingData) = Simulation.Utilities.InitializeSimulation1D(
                steps,
                cycles,
                min[0],
                max[0],
                numNeurons,
                placeFieldRadius,
                firingThreshold
            );

            RunSortedSpikeEncoder1D(
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
            var (position2D, spikingData) = Simulation.Utilities.InitializeSimulation2D(
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

            RunSortedSpikeEncoder2D(
                sortedSpikeEncoder,
                evaluationSteps,
                position2D, 
                spikingData, 
                outputDirectory,
                [ min[0], max[0], min[1], max[1] ]
            );
        }
    }

    public static void ClusterlessMarkEncoder(
        EstimationMethod estimationMethod = EstimationMethod.KernelDensity,
        double[]? covariateBandwidth = null,
        int numDimensions = 1,
        long[]? evaluationSteps = null,
        int steps = 200,
        int cycles = 10,
        double[]? min = null,
        double[]? max = null,
        double? distanceThreshold = null,
        int markDimensions = 4,
        int numChannels = 8,
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
        covariateBandwidth ??= [5.0];
        evaluationSteps ??= [50];
        markBandwidth ??= [1.0, 1.0, 1.0, 1.0];
        min ??= [0.0];
        max ??= [100.0];
        device ??= CPU;

        outputDirectory = Path.Combine(outputDirectory, modelDirectory, estimationMethod.ToString());

        var stateSpace = new DiscreteUniform(
            numDimensions,
            min,
            max,
            evaluationSteps,
            device: device,
            scalarType: scalarType
        );

        var encoder = new ClusterlessMarks(
            estimationMethod, 
            covariateBandwidth,
            markDimensions,
            numChannels,
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
            (position, spikingData) = Simulation.Utilities.InitializeSimulation1D(
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
            (position, spikingData) = Simulation.Utilities.InitializeSimulation2D(
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
            numChannels,
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
            plotDensity2D.Show(density);
            plotDensity2D.Save(png: true);
        }
    }

    public static void RunSortedSpikeEncoder1D(
        IEncoder encoder, 
        Tensor covariates, 
        Tensor spikes,
        string encoderDirectory,
        long evaluationSteps,
        double[] densityScatterPlotRange,
        double[] densityHeatmapRange,
        int heatmapPadding,
        string title = ""
    )
    {
        encoder.Encode(covariates, spikes);
        var densities = encoder.Evaluate().First();

        for (int i = 0; i < densities.shape[0]; i++)
        {
            var density = densities[i]
                .exp();
            var density1DExpanded = vstack([arange(evaluationSteps), density]).T;

            var directoryScatterPlot1D = Path.Combine(encoderDirectory, "ScatterPlot1D");

            ScatterPlot plotDensity1D = new ScatterPlot(
                densityScatterPlotRange[0],
                densityScatterPlotRange[1],
                densityScatterPlotRange[2],
                densityScatterPlotRange[3],
                title: $"{title}{i}"
            );
            plotDensity1D.OutputDirectory = Path.Combine(plotDensity1D.OutputDirectory, directoryScatterPlot1D);
            plotDensity1D.Show(density1DExpanded);
            plotDensity1D.Save(png: true);

            var density2D = tile(density, [heatmapPadding, 1]);

            var directoryHeatmap2D = Path.Combine(encoderDirectory, "Heatmap2D");

            Heatmap plotDensity2D = new(
                densityHeatmapRange[0],
                densityHeatmapRange[1],
                densityHeatmapRange[2],
                densityHeatmapRange[3],
                title: $"{title}{i}"
            );
            plotDensity2D.OutputDirectory = Path.Combine(plotDensity2D.OutputDirectory, directoryHeatmap2D);
            plotDensity2D.Show(density2D);
            plotDensity2D.Save(png: true);
        }
    }

    public static void RunSortedSpikeEncoder2D(
        IEncoder encoder,
        long[] evaluationSteps,
        Tensor covariates, 
        Tensor spikes,
        string encoderDirectory,
        double[] densityHeatmapRange,
        string title = ""
    )
    {
        encoder.Encode(covariates, spikes);
        var densities = encoder.Evaluate().First();

        for (int i = 0; i < densities.shape[0]; i++)
        {
            var density = densities[i]
                .exp()
                .reshape(evaluationSteps);
            var directoryHeatmap2D = Path.Combine(encoderDirectory, "Heatmap2D");

            Heatmap plotDensity2D = new(
                densityHeatmapRange[0],
                densityHeatmapRange[1],
                densityHeatmapRange[2],
                densityHeatmapRange[3],
                title: $"{title}{i}"
            );

            plotDensity2D.OutputDirectory = Path.Combine(plotDensity2D.OutputDirectory, directoryHeatmap2D);
            plotDensity2D.Show(density);
            plotDensity2D.Save(png: true);
        }
    }
}