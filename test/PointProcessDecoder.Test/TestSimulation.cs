using static TorchSharp.torch;
using PointProcessDecoder.Core;
using PointProcessDecoder.Plot;
using PointProcessDecoder.Simulation;

namespace PointProcessDecoder.Test;

[TestClass]
public class TestSimulation
{
    private readonly int seed = 0;
    private readonly ScalarType scalarType = ScalarType.Float32;
    private readonly int numNeurons = 40;
    private readonly double placeFieldRadius = 8.0;
    private readonly double firingThreshold = 0.2;
    private readonly string outputDirectory = "TestSimulation";

    [TestMethod]
    public void TestSpikingNeurons1D()
    {
        var steps = 200;
        var cycles = 10;
        var min = 0.0;
        var max = 100.0;

        var spikingNeuronsDirectory = Path.Combine(outputDirectory, "SpikingNeurons1D");

        var position1D = Simulate.Position(steps, cycles, min, max, scalarType);
        var position1DExpanded = concat([zeros_like(position1D), position1D], dim: 1);
        var position1DExpandedTime = concat([arange(position1D.shape[0]).unsqueeze(1), position1D], dim: 1);
        var minPosition = 0;
        var maxPosition = position1D.shape[0];
        
        ScatterPlot plotPosition1D = new(minPosition, maxPosition, min, max, "Position1D");
        plotPosition1D.OutputDirectory = Path.Combine(plotPosition1D.OutputDirectory, spikingNeuronsDirectory);
        plotPosition1D.Show<float>(position1DExpandedTime);
        plotPosition1D.Save(png: true);

        var placeFieldCenters = Simulate.PlaceFieldCenters(min, max, numNeurons, seed, scalarType);
        var placeFieldCenters2D = concat([zeros_like(placeFieldCenters), placeFieldCenters], dim: 1);

        ScatterPlot plotPlaceFieldCenters = new(-1, 1, min, max, "PlaceFieldCenters1D");
        plotPlaceFieldCenters.OutputDirectory = Path.Combine(plotPlaceFieldCenters.OutputDirectory, spikingNeuronsDirectory);
        plotPlaceFieldCenters.Show<float>(placeFieldCenters2D);
        plotPlaceFieldCenters.Save(png: true);

        var spikingData = Simulate.SpikesAtPosition(position1D, placeFieldCenters, placeFieldRadius, firingThreshold, seed);

        ScatterPlot plotSpikingNeurons = new(0, position1D.shape[0], min, max, title: "SpikingNeurons1D");
        plotSpikingNeurons.OutputDirectory = Path.Combine(plotSpikingNeurons.OutputDirectory, spikingNeuronsDirectory);

        var colors = Utilities.GenerateRandomColors(numNeurons, seed);

        for (int i = 0; i < numNeurons; i++)
        {
            var positionsAtSpikes = position1DExpandedTime[spikingData[TensorIndex.Ellipsis, i]];
            plotSpikingNeurons.Show<float>(positionsAtSpikes, colors[i]);
        }
        plotSpikingNeurons.Save(png: true);
    }

    [TestMethod]
    public void TestSpikingNeurons2D()
    {
        var steps = 200;
        var cycles = 10;
        var xMin = 0.0;
        var xMax = 100.0;
        var yMin = 0.0;
        var yMax = 100.0;

        var spikingNeuronsDirectory = Path.Combine(outputDirectory, "SpikingNeurons2D");

        var position2D = Simulate.Position(steps, cycles, xMin, xMax, yMin, yMax, scalarType: scalarType);
        
        ScatterPlot plotPosition2D = new(xMin, xMax, yMin, yMax, "Position2D");
        plotPosition2D.OutputDirectory = Path.Combine(plotPosition2D.OutputDirectory, spikingNeuronsDirectory);
        plotPosition2D.Show<float>(position2D);
        plotPosition2D.Save(png: true);

        var placeFieldCenters = Simulate.PlaceFieldCenters(xMin, xMax, yMin, yMax, numNeurons, seed, scalarType);

        ScatterPlot plotPlaceFieldCenters = new(xMin, xMax, yMin, yMax, "PlaceFieldCenters");
        plotPlaceFieldCenters.OutputDirectory = Path.Combine(plotPlaceFieldCenters.OutputDirectory, spikingNeuronsDirectory);
        plotPlaceFieldCenters.Show<float>(placeFieldCenters);
        plotPlaceFieldCenters.Save(png: true);

        var spikingData = Simulate.SpikesAtPosition(position2D, placeFieldCenters, placeFieldRadius, firingThreshold, seed);

        ScatterPlot plotSpikingNeurons = new(xMin, xMax, yMin, yMax, title: "SpikingNeurons");
        plotSpikingNeurons.OutputDirectory = Path.Combine(plotSpikingNeurons.OutputDirectory, spikingNeuronsDirectory);

        var colors = Utilities.GenerateRandomColors(numNeurons, seed);

        for (int i = 0; i < numNeurons; i++)
        {
            var positionsAtSpikes = position2D[spikingData[TensorIndex.Colon, i]];
            plotSpikingNeurons.Show<float>(positionsAtSpikes, colors[i]);
        }
        plotSpikingNeurons.Save(png: true);
    }
}