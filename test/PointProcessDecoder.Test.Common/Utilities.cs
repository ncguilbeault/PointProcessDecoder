using static TorchSharp.torch;
using PointProcessDecoder.Simulation;
using PointProcessDecoder.Plot;
using PointProcessDecoder.Core;

namespace PointProcessDecoder.Test.Common;

public static class Utilities
{
    public static void RunSortedSpikeEncoder1D(
        IEncoder encoder, 
        Tensor observations, 
        Tensor spikes,
        string encoderDirectory,
        long evaluationSteps,
        double[] densityScatterPlotRange,
        double[] densityHeatmapRange,
        int heatmapPadding,
        string title = ""
    )
    {
        encoder.Encode(observations, spikes);
        var densities = encoder.Evaluate().First();

        for (int i = 0; i < densities.shape[0]; i++)
        {
            var density = densities[i];
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
            plotDensity1D.Show<float>(density1DExpanded);
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
            plotDensity2D.Show<float>(density2D);
            plotDensity2D.Save(png: true);
        }
    }

    public static void RunSortedSpikeEncoder2D(

        IEncoder encoder, 
        Tensor observations, 
        Tensor spikes,
        string encoderDirectory,
        double[] densityHeatmapRange,
        string title = ""
    )
    {
        encoder.Encode(observations, spikes);
        var densities = encoder.Evaluate().First();

        for (int i = 0; i < densities.shape[0]; i++)
        {
            var density = densities[i];
            var directoryHeatmap2D = Path.Combine(encoderDirectory, "Heatmap2D");

            Heatmap plotDensity2D = new(
                densityHeatmapRange[0],
                densityHeatmapRange[1],
                densityHeatmapRange[2],
                densityHeatmapRange[3],
                title: $"{title}{i}"
            );

            plotDensity2D.OutputDirectory = Path.Combine(plotDensity2D.OutputDirectory, directoryHeatmap2D);
            plotDensity2D.Show<float>(density);
            plotDensity2D.Save(png: true);
        }
    }

    public static Tensor ReadBinaryFile(
        string binary_file,
        Device? device = null,
        ScalarType scalarType = ScalarType.Float32
    )
    {
        device ??= CPU;
        byte[] fileBytes = File.ReadAllBytes(binary_file);
        int elementCount = fileBytes.Length / sizeof(double);
        double[] doubleArray = new double[elementCount];
        Buffer.BlockCopy(fileBytes, 0, doubleArray, 0, fileBytes.Length);
        Tensor t = tensor(doubleArray, device: device, dtype: scalarType);
        return t;
    }

    public static (Tensor, Tensor) InitializeRealClusterlessMarksData(
        string positionFile,
        string marksFile,
        Device? device = null,
        ScalarType scalarType = ScalarType.Float32
    )
    {
        var position = ReadBinaryFile(positionFile, device, scalarType);
        var marks = ReadBinaryFile(marksFile, device, scalarType);
        return (position, marks);
    }

    public static (Tensor, Tensor) InitializeRealSortedSpikeData(
        string positionFile,
        string spikesFile,
        Device? device = null,
        ScalarType scalarType = ScalarType.Float32
    )
    {
        var position = ReadBinaryFile(positionFile, device, scalarType);
        var spikes = ReadBinaryFile(spikesFile, device, scalarType);
        return (position, spikes);
    }

    public static (Tensor, Tensor) InitializeSimulation1D(
        int steps = 200,
        int cycles = 10,
        double min = 0.0,
        double max = 100.0,
        int numNeurons = 40,
        double placeFieldRadius = 8.0,
        double firingThreshold = 0.2,
        ScalarType scalarType = ScalarType.Float32,
        Device? device = null,
        int seed = 0
    )
    {
        var position1D = Simulate.Position(
            steps, 
            cycles, 
            min, 
            max, 
            scalarType,
            device
        );
        
        var position1DExpanded = concat([zeros_like(position1D), position1D], dim: 1);

        var placeFieldCenters = Simulate.PlaceFieldCenters(
            min, 
            max, 
            numNeurons, 
            seed, 
            scalarType,
            device
        );

        var placeFieldCenters2D = concat([zeros_like(placeFieldCenters), placeFieldCenters], dim: 1);

        var spikingData = Simulate.SpikesAtPosition(
            position1DExpanded, 
            placeFieldCenters2D, 
            placeFieldRadius, 
            firingThreshold, 
            seed,
            device: device
        );

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
        Device? device = null,
        int seed = 0
    )
    {
        var position2D = Simulate.Position(
            steps, 
            cycles, 
            xMin, 
            xMax, 
            yMin, 
            yMax, 
            scale: scale, 
            scalarType: scalarType,
            device: device
        );

        var placeFieldCenters = Simulate.PlaceFieldCenters(
            xMin, 
            yMax, 
            yMin, 
            yMax, 
            numNeurons, 
            seed, 
            scalarType,
            device
        );

        var spikingData = Simulate.SpikesAtPosition(
            position2D, 
            placeFieldCenters, 
            placeFieldRadius, 
            firingThreshold, 
            seed,
            device: device
        );

        return (position2D, spikingData);
    }

}
