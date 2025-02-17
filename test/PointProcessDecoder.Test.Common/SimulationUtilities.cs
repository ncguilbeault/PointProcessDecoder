using static TorchSharp.torch;
using PointProcessDecoder.Plot;
using PointProcessDecoder.Simulation;

namespace PointProcessDecoder.Test.Common;

public static class SimulationUtilities
{
    public static void SpikingNeurons1D(
        int steps = 200,
        int cycles = 10,
        double min = 0.0,
        double max = 100.0,
        int seed = 0,
        ScalarType scalarType = ScalarType.Float32,
        Device? device = null,
        int numNeurons = 40,
        double placeFieldRadius = 8.0,
        double firingThreshold = 0.2,
        string outputDirectory = "TestSimulation",
        string testDirectory = "SpikingNeurons1D"
    )
    {
        device ??= CPU;

        outputDirectory = string.IsNullOrEmpty(testDirectory) ? outputDirectory : Path.Combine(outputDirectory, testDirectory);

        var position1D = Simulate.Position(
            steps, 
            cycles, 
            min, 
            max, 
            scalarType,
            device
        );
        var position1DExpandedTime = concat([arange(position1D.shape[0]).unsqueeze(1), position1D], dim: 1);
        var minPosition = 0;
        var maxPosition = position1D.shape[0];
        
        ScatterPlot plotPosition1D = new(
            xMin: minPosition, 
            xMax: maxPosition, 
            yMin: min, 
            yMax: max, 
            title: "Position1D"
        );

        plotPosition1D.OutputDirectory = Path.Combine(plotPosition1D.OutputDirectory, outputDirectory);
        plotPosition1D.Show<float>(position1DExpandedTime);
        plotPosition1D.Save(png: true);

        var placeFieldCenters = Simulate.PlaceFieldCenters(
            min, 
            max, 
            numNeurons, 
            seed, 
            scalarType,
            device
        );
        var placeFieldCenters2D = concat([zeros_like(placeFieldCenters), placeFieldCenters], dim: 1);

        ScatterPlot plotPlaceFieldCenters = new(
            xMin: -1, 
            xMax: 1, 
            yMin: min, 
            yMax: max, 
            title: "PlaceFieldCenters1D"
        );

        plotPlaceFieldCenters.OutputDirectory = Path.Combine(plotPlaceFieldCenters.OutputDirectory, outputDirectory);
        plotPlaceFieldCenters.Show<float>(placeFieldCenters2D);
        plotPlaceFieldCenters.Save(png: true);

        var spikingData = Simulate.SpikesAtPosition(
            position1D, 
            placeFieldCenters,
            placeFieldRadius, 
            firingThreshold, 
            seed,
            device: device
        );

        ScatterPlot plotSpikingNeurons = new(
            xMin: 0, 
            xMax: position1D.shape[0], 
            yMin: min, 
            yMax: max, 
            title: "SpikingNeurons1D"
        );

        plotSpikingNeurons.OutputDirectory = Path.Combine(plotSpikingNeurons.OutputDirectory, outputDirectory);

        var colors = Plot.Utilities.GenerateRandomColors(numNeurons, seed);

        for (int i = 0; i < numNeurons; i++)
        {
            var spikesMask = spikingData[TensorIndex.Ellipsis, i] != 0;
            var positionsAtSpikes = position1DExpandedTime[spikesMask];
            plotSpikingNeurons.Show<float>(positionsAtSpikes, colors[i]);
        }
        plotSpikingNeurons.Save(png: true);
    }

    public static void SpikingNeurons2D(
        int steps = 200,
        int cycles = 10,
        double xMin = 0.0,
        double xMax = 100.0,
        double yMin = 0.0,
        double yMax = 100.0,
        int seed = 0,
        ScalarType scalarType = ScalarType.Float32,
        Device? device = null,
        int numNeurons = 40,
        double placeFieldRadius = 8.0,
        double firingThreshold = 0.2,
        string outputDirectory = "TestSimulation",
        string testDirectory = "SpikingNeurons2D"
    )
    {
        device ??= CPU;
        outputDirectory = string.IsNullOrEmpty(testDirectory) ? outputDirectory : Path.Combine(outputDirectory, testDirectory);

        var position2D = Simulate.Position(
            steps, 
            cycles, 
            xMin, 
            xMax, 
            yMin, 
            yMax, 
            scalarType: scalarType,
            device: device
        );
        
        ScatterPlot plotPosition2D = new(
            xMin: xMin, 
            xMax: xMax, 
            yMin: yMin, 
            yMax: yMax, 
            title: "Position2D"
        );

        plotPosition2D.OutputDirectory = Path.Combine(plotPosition2D.OutputDirectory, outputDirectory);
        plotPosition2D.Show<float>(position2D);
        plotPosition2D.Save(png: true);

        var placeFieldCenters = Simulate.PlaceFieldCenters(
            xMin,
            xMax,
            yMin,
            yMax,
            numNeurons,
            seed,
            scalarType,
            device: device
        );

        ScatterPlot plotPlaceFieldCenters = new(
            xMin: xMin, 
            xMax: xMax, 
            yMin: yMin, 
            yMax: yMax, 
            title: "PlaceFieldCenters"
        );

        plotPlaceFieldCenters.OutputDirectory = Path.Combine(plotPlaceFieldCenters.OutputDirectory, outputDirectory);
        plotPlaceFieldCenters.Show<float>(placeFieldCenters);
        plotPlaceFieldCenters.Save(png: true);

        var spikingData = Simulate.SpikesAtPosition(
            position2D, 
            placeFieldCenters, 
            placeFieldRadius, 
            firingThreshold, 
            seed,
            device: device
        );

        ScatterPlot plotSpikingNeurons = new(
            xMin: xMin, 
            xMax: xMax, 
            yMin: yMin, 
            yMax: yMax, 
            title: "SpikingNeurons"
        );

        plotSpikingNeurons.OutputDirectory = Path.Combine(plotSpikingNeurons.OutputDirectory, outputDirectory);

        var colors = Plot.Utilities.GenerateRandomColors(numNeurons, seed);

        for (int i = 0; i < numNeurons; i++)
        {
            var spikesMask = spikingData[TensorIndex.Ellipsis, i] != 0;
            var positionsAtSpikes = position2D[spikesMask];
            plotSpikingNeurons.Show<float>(positionsAtSpikes, colors[i]);
        }
        plotSpikingNeurons.Save(png: true);
    }

    public static void SpikingNeurons2DFirstAndLastSteps(
        int steps = 200,
        int cycles = 10,
        double scale = 0.1,
        double xMin = 0.0,
        double xMax = 100.0,
        double yMin = 0.0,
        double yMax = 100.0,
        int stepsToSeperate = 1800,
        int seed = 0,
        ScalarType scalarType = ScalarType.Float32,
        Device? device = null,
        int numNeurons = 40,
        double placeFieldRadius = 8.0,
        double firingThreshold = 0.2,
        string outputDirectory = "TestSimulation",
        string testDirectory = "SpikingNeurons2DFirstAndLastSteps"
    )
    {
        device ??= CPU;
        outputDirectory = string.IsNullOrEmpty(testDirectory) ? outputDirectory : Path.Combine(outputDirectory, testDirectory);

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
        
        ScatterPlot plotPositionFirst = new(
            xMin: xMin, 
            xMax: xMax, 
            yMin: yMin, 
            yMax: yMax, 
            title: "Position2DFirst"
        );

        plotPositionFirst.OutputDirectory = Path.Combine(plotPositionFirst.OutputDirectory, outputDirectory);
        plotPositionFirst.Show<float>(position2D[TensorIndex.Slice(0, stepsToSeperate)]);
        plotPositionFirst.Save(png: true);

        ScatterPlot plotPositionLast = new(
            xMin: xMin, 
            xMax: xMax, 
            yMin: yMin, 
            yMax: yMax, 
            title: "Position2DLast"
        );

        plotPositionLast.OutputDirectory = Path.Combine(plotPositionLast.OutputDirectory, outputDirectory);
        plotPositionLast.Show<float>(position2D[TensorIndex.Slice(stepsToSeperate)]);
        plotPositionLast.Save(png: true);

        var placeFieldCenters = Simulate.PlaceFieldCenters(
            xMin, 
            xMax, 
            yMin, 
            yMax, 
            numNeurons, 
            seed, 
            scalarType,
            device: device
        );

        ScatterPlot plotPlaceFieldCenters = new(
            xMin: xMin, 
            xMax: xMax, 
            yMin: yMin, 
            yMax: yMax, 
            title: "PlaceFieldCenters"
        );

        plotPlaceFieldCenters.OutputDirectory = Path.Combine(plotPlaceFieldCenters.OutputDirectory, outputDirectory);
        plotPlaceFieldCenters.Show<float>(placeFieldCenters);
        plotPlaceFieldCenters.Save(png: true);

        var spikingData = Simulate.SpikesAtPosition(
            position2D, 
            placeFieldCenters, 
            placeFieldRadius, 
            firingThreshold, 
            seed,
            device: device
        );

        ScatterPlot plotSpikingNeuronsFirst = new(
            xMin: xMin, 
            xMax: xMax, 
            yMin: yMin, 
            yMax: yMax, 
            title: "SpikingNeuronsFirst"
        );

        plotSpikingNeuronsFirst.OutputDirectory = Path.Combine(plotSpikingNeuronsFirst.OutputDirectory, outputDirectory);

        ScatterPlot plotSpikingNeuronsLast = new(
            xMin: xMin, 
            xMax: xMax, 
            yMin: yMin, 
            yMax: yMax, 
            title: "SpikingNeuronsLast"
        );

        plotSpikingNeuronsLast.OutputDirectory = Path.Combine(plotSpikingNeuronsLast.OutputDirectory, outputDirectory);

        var colors = Plot.Utilities.GenerateRandomColors(numNeurons, seed);

        for (int i = 0; i < numNeurons; i++)
        {
            var spikesMaskFirst = spikingData[TensorIndex.Slice(0, stepsToSeperate), i] != 0;
            var positionsAtSpikesFirst = position2D[TensorIndex.Slice(0, stepsToSeperate)][spikesMaskFirst];
            plotSpikingNeuronsFirst.Show<float>(positionsAtSpikesFirst, colors[i]);

            var spikesMaskLast = spikingData[TensorIndex.Slice(stepsToSeperate), i] != 0;
            var positionsAtSpikesLast = position2D[TensorIndex.Slice(stepsToSeperate)][spikesMaskLast];
            plotSpikingNeuronsLast.Show<float>(positionsAtSpikesLast, colors[i]);
        }

        plotSpikingNeuronsFirst.Save(png: true);
        plotSpikingNeuronsLast.Save(png: true);
    }

    public static void Marks1D(
        int steps = 200,
        int cycles = 10,
        double min = 0.0,
        double max = 100.0,
        int stepsToSeperate = 1800,
        int seed = 0,
        ScalarType scalarType = ScalarType.Float32,
        Device? device = null,
        int numNeurons = 40,
        double placeFieldRadius = 8.0,
        double firingThreshold = 0.2,
        int markDimensions = 4,
        int markChannels = 8,
        double spikeScale = 5,
        double noiseScale = 0.5,
        string outputDirectory = "TestSimulation",
        string testDirectory = "Marks1D"
    )
    {
        device ??= CPU;
        outputDirectory = string.IsNullOrEmpty(testDirectory) ? outputDirectory : Path.Combine(outputDirectory, testDirectory);

        var position = Simulate.Position(
            steps, 
            cycles, 
            min, 
            max,
            scalarType: scalarType,
            device: device
        );

        var placeFieldCenters = Simulate.PlaceFieldCenters(
            min, 
            max,
            numNeurons, 
            seed, 
            scalarType,
            device: device
        );

        var spikingData = Simulate.SpikesAtPosition(
            position, 
            placeFieldCenters, 
            placeFieldRadius, 
            firingThreshold, 
            seed,
            device: device
        );

        var marks = Simulate.MarksAtPosition(
            position,
            spikingData, 
            markDimensions, 
            markChannels, 
            seed, 
            device: device,
            spikeScale: spikeScale,
            noiseScale: noiseScale
        );

        var marksMin = marks.nan_to_num().min().item<float>();
        var marksMax = marks.nan_to_num().max().item<float>();

        for (int i = 0; i < markChannels; i++)
        {
            var marksChannel = marks[TensorIndex.Ellipsis, i];
            var marksMasked = marksChannel.isnan().logical_not().all(1);
            for (int j = 0; j < markDimensions - 1; j++)
            {
                var marksDim1 = marksChannel[marksMasked, j];
                var marksDim2 = marksChannel[marksMasked, j + 1];
                var marksAtSpikes = stack([marksDim1, marksDim2], dim: 1);
                ScatterPlot plotMarks = new(marksMin, marksMax, marksMin, marksMax, title: $"Marks_{i}_{j}");
                plotMarks.OutputDirectory = Path.Combine(plotMarks.OutputDirectory, outputDirectory);
                plotMarks.Show<float>(marksAtSpikes);
                plotMarks.Save(png: true);
            }
        }

    }
}