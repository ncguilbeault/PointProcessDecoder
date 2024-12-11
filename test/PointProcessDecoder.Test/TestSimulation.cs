using static TorchSharp.torch;
using PointProcessDecoder.Core;
using PointProcessDecoder.Plot;
using PointProcessDecoder.Simulation;

namespace PointProcessDecoder.Test;

[TestClass]
public class TestSimulation
{
    [TestMethod]
    public void TestSpikingNeurons()
    {
        var seed = 0;
        var steps = 200;
        var cycles = 10;
        var yMin = 0.0;
        var yMax = 100.0;
        var scalarType = ScalarType.Float64;

        var spikingNeuronsDirectory = "TestSimulation";

        var position1D = Simulate.Position(steps, cycles, yMin, yMax, scalarType);
        var position1DExpanded = concat([zeros_like(position1D), position1D], dim: 1);
        var position1DExpandedTime = concat([arange(position1D.shape[0]).unsqueeze(1), position1D], dim: 1);
        var minPosition = 0;
        var maxPosition = position1D.shape[0];
        
        ScatterPlot plotPosition1D = new(minPosition, maxPosition, yMin, yMax, "Position1D");
        plotPosition1D.OutputDirectory = Path.Combine(plotPosition1D.OutputDirectory, spikingNeuronsDirectory);
        plotPosition1D.Show(position1DExpandedTime);
        plotPosition1D.Save(png: true);

        var numNeurons = 40;

        var placeFieldCenters = Simulate.PlaceFieldCenters(yMin, yMax, numNeurons, seed, scalarType);
        var placeFieldCenters2D = vstack([zeros_like(placeFieldCenters), placeFieldCenters]).T;

        ScatterPlot plotPlaceFieldCenters = new(-1, 1, yMin, yMax, "PlaceFieldCenters");
        plotPlaceFieldCenters.OutputDirectory = Path.Combine(plotPlaceFieldCenters.OutputDirectory, spikingNeuronsDirectory);
        plotPlaceFieldCenters.Show(placeFieldCenters2D);
        plotPlaceFieldCenters.Save(png: true);

        var placeFieldRadius = 8.0;
        var firingThreshold = 0.2;

        var spikingData = Simulate.SpikesAtPosition(position1DExpanded, placeFieldCenters2D, placeFieldRadius, firingThreshold, seed);

        ScatterPlot plotSpikingNeurons = new(0, position1D.shape[0], yMin, yMax, title: "SpikingNeurons");
        plotSpikingNeurons.OutputDirectory = Path.Combine(plotSpikingNeurons.OutputDirectory, spikingNeuronsDirectory);

        var colors = Utilities.GenerateRandomColors(numNeurons);

        for (int i = 0; i < numNeurons; i++)
        {
            var positionsAtSpikes = position1DExpandedTime[spikingData[TensorIndex.Colon, i]];
            plotSpikingNeurons.Show(positionsAtSpikes, colors[i]);
        }
        plotSpikingNeurons.Save(png: true);

        Assert.AreEqual(numNeurons, colors.Count);
    }
}