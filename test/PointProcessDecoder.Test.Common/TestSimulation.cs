using static TorchSharp.torch;
using PointProcessDecoder.Plot;
using PointProcessDecoder.Simulation;

namespace PointProcessDecoder.Test.Common;

public static class TestSimulation
{
    public static void TestSpikingNeurons1D(
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
        string outputDirectory = "TestSimulation"
    )
    {
        device ??= CPU;

        var spikingNeuronsDirectory = Path.Combine(outputDirectory, "SpikingNeurons1D");

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
        
        ScatterPlot plotPosition1D = new(minPosition, maxPosition, min, max, "Position1D");
        plotPosition1D.OutputDirectory = Path.Combine(plotPosition1D.OutputDirectory, spikingNeuronsDirectory);
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

        ScatterPlot plotPlaceFieldCenters = new(-1, 1, min, max, "PlaceFieldCenters1D");
        plotPlaceFieldCenters.OutputDirectory = Path.Combine(plotPlaceFieldCenters.OutputDirectory, spikingNeuronsDirectory);
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

        ScatterPlot plotSpikingNeurons = new(0, position1D.shape[0], min, max, title: "SpikingNeurons1D");
        plotSpikingNeurons.OutputDirectory = Path.Combine(plotSpikingNeurons.OutputDirectory, spikingNeuronsDirectory);

        var colors = Plot.Utilities.GenerateRandomColors(numNeurons, seed);

        for (int i = 0; i < numNeurons; i++)
        {
            var positionsAtSpikes = position1DExpandedTime[spikingData[TensorIndex.Ellipsis, i]];
            plotSpikingNeurons.Show<float>(positionsAtSpikes, colors[i]);
        }
        plotSpikingNeurons.Save(png: true);
    }

    public static void TestSpikingNeurons2D(
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
        string outputDirectory = "TestSimulation"
    )
    {
        device ??= CPU;
        var spikingNeuronsDirectory = Path.Combine(outputDirectory, "SpikingNeurons2D");

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
        
        ScatterPlot plotPosition2D = new(xMin, xMax, yMin, yMax, "Position2D");
        plotPosition2D.OutputDirectory = Path.Combine(plotPosition2D.OutputDirectory, spikingNeuronsDirectory);
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

        ScatterPlot plotPlaceFieldCenters = new(xMin, xMax, yMin, yMax, "PlaceFieldCenters");
        plotPlaceFieldCenters.OutputDirectory = Path.Combine(plotPlaceFieldCenters.OutputDirectory, spikingNeuronsDirectory);
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

        ScatterPlot plotSpikingNeurons = new(xMin, xMax, yMin, yMax, title: "SpikingNeurons");
        plotSpikingNeurons.OutputDirectory = Path.Combine(plotSpikingNeurons.OutputDirectory, spikingNeuronsDirectory);

        var colors = Plot.Utilities.GenerateRandomColors(numNeurons, seed);

        for (int i = 0; i < numNeurons; i++)
        {
            var positionsAtSpikes = position2D[spikingData[TensorIndex.Colon, i]];
            plotSpikingNeurons.Show<float>(positionsAtSpikes, colors[i]);
        }
        plotSpikingNeurons.Save(png: true);
    }

    public static void TestSpikingNeurons2DFirstAndLastSteps(
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
        string outputDirectory = "TestSimulation"
    )
    {
        device ??= CPU;

        var spikingNeuronsDirectory = Path.Combine(outputDirectory, "SpikingNeurons2DFirstAndLastSteps");

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
        
        ScatterPlot plotPositionFirst = new(xMin, xMax, yMin, yMax, "Position2DFirst");
        plotPositionFirst.OutputDirectory = Path.Combine(plotPositionFirst.OutputDirectory, spikingNeuronsDirectory);
        plotPositionFirst.Show<float>(position2D[TensorIndex.Slice(0, stepsToSeperate)]);
        plotPositionFirst.Save(png: true);

        ScatterPlot plotPositionLast = new(xMin, xMax, yMin, yMax, "Position2DLast");
        plotPositionLast.OutputDirectory = Path.Combine(plotPositionLast.OutputDirectory, spikingNeuronsDirectory);
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

        ScatterPlot plotPlaceFieldCenters = new(xMin, xMax, yMin, yMax, "PlaceFieldCenters");
        plotPlaceFieldCenters.OutputDirectory = Path.Combine(plotPlaceFieldCenters.OutputDirectory, spikingNeuronsDirectory);
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

        ScatterPlot plotSpikingNeuronsFirst = new(xMin, xMax, yMin, yMax, title: "SpikingNeuronsFirst");
        plotSpikingNeuronsFirst.OutputDirectory = Path.Combine(plotSpikingNeuronsFirst.OutputDirectory, spikingNeuronsDirectory);

        ScatterPlot plotSpikingNeuronsLast = new(xMin, xMax, yMin, yMax, title: "SpikingNeuronsLast");
        plotSpikingNeuronsLast.OutputDirectory = Path.Combine(plotSpikingNeuronsLast.OutputDirectory, spikingNeuronsDirectory);

        var colors = Plot.Utilities.GenerateRandomColors(numNeurons, seed);

        for (int i = 0; i < numNeurons; i++)
        {
            var positionsAtSpikesFirst = position2D[TensorIndex.Slice(0, stepsToSeperate)][spikingData[TensorIndex.Slice(0, stepsToSeperate), i]];
            plotSpikingNeuronsFirst.Show<float>(positionsAtSpikesFirst, colors[i]);

            var positionsAtSpikesLast = position2D[TensorIndex.Slice(stepsToSeperate)][spikingData[TensorIndex.Slice(stepsToSeperate), i]];
            plotSpikingNeuronsLast.Show<float>(positionsAtSpikesLast, colors[i]);
        }

        plotSpikingNeuronsFirst.Save(png: true);
        plotSpikingNeuronsLast.Save(png: true);
    }
}