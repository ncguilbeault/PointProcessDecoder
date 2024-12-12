using static TorchSharp.torch;
using PointProcessDecoder.Core;
using PointProcessDecoder.Plot;
using PointProcessDecoder.Simulation;
using PointProcessDecoder.Core.Estimation;
using PointProcessDecoder.Core.Transitions;
using PointProcessDecoder.Core.Encoder;
using PointProcessDecoder.Core.Decoder;
using System.Text;

namespace PointProcessDecoder.Test;

[TestClass]
public class TestModel
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
    private double bandwidth = 10.0;
    private int numDimensions = 1;
    private long evaluationSteps = 50;
    private double sigma = 0.1;
    private double distanceThreshold = 1.5;
    private int testingSteps = 1800;
    private string outputDirectory = "TestModel";

    private (Tensor, Tensor) InitializeSimulation1D()
    {
        var position1D = Simulate.Position(steps, cycles, yMin, yMax, scalarType);
        var position1DExpanded = concat([zeros_like(position1D), position1D], dim: 1);

        var placeFieldCenters = Simulate.PlaceFieldCenters(yMin, yMax, numNeurons, seed, scalarType);
        var placeFieldCenters2D = vstack([zeros_like(placeFieldCenters), placeFieldCenters]).T;

        var spikingData = Simulate.SpikesAtPosition(position1DExpanded, placeFieldCenters2D, placeFieldRadius, firingThreshold, seed);

        return (position1D, spikingData);
    }

    [TestMethod]
    public void TestPointProcessModelUniformTransitions()
    {
        var pointProcessModelDirectory = Path.Combine(outputDirectory, "PointProcessModelUniformTransitions");
        var (position1D, spikingData) = InitializeSimulation1D();

        var pointProcessModel = new PointProcessModel(
            EstimationMethod.KernelDensity,
            TransitionsType.Uniform,
            EncoderType.SortedSpikeEncoder,
            DecoderType.SortedSpikeDecoder,
            [yMin],
            [yMax],
            [evaluationSteps],
            [bandwidth],
            latentDimensions: numDimensions,
            nUnits: numNeurons,
            device: device
        );

        pointProcessModel.Encode(position1D[TensorIndex.Slice(0, testingSteps)], spikingData[TensorIndex.Slice(0, testingSteps)]);
        var prediction = pointProcessModel.Decode(spikingData[TensorIndex.Slice(testingSteps)]);

        Heatmap plotPrediction2D = new(
            0,
            steps * cycles - testingSteps,
            yMin,
            yMax,
            title: $"Prediction2D"
        );

        plotPrediction2D.OutputDirectory = Path.Combine(plotPrediction2D.OutputDirectory, pointProcessModelDirectory);
        plotPrediction2D.Show<float>(prediction);
        plotPrediction2D.Save(png: true);

        Assert.IsTrue(true);
    }

    [TestMethod]
    public void TestPointProcessModelRandomWalkTransitions()
    {
        var pointProcessModelDirectory = Path.Combine(outputDirectory, "PointProcessModelRandomWalkTransitions");
        var (position1D, spikingData) = InitializeSimulation1D();

        var pointProcessModel = new PointProcessModel(
            EstimationMethod.KernelDensity,
            TransitionsType.RandomWalk,
            EncoderType.SortedSpikeEncoder,
            DecoderType.SortedSpikeDecoder,
            [yMin],
            [yMax],
            [evaluationSteps],
            [bandwidth],
            latentDimensions: numDimensions,
            nUnits: numNeurons,
            sigmaLatentSpace: [sigma],
            device: device
        );

        pointProcessModel.Encode(position1D[TensorIndex.Slice(0, testingSteps)], spikingData[TensorIndex.Slice(0, testingSteps)]);
        var prediction = pointProcessModel.Decode(spikingData[TensorIndex.Slice(testingSteps)]);

        Heatmap plotPrediction2D = new(
            0,
            steps * cycles - testingSteps,
            yMin,
            yMax,
            title: $"Prediction2D"
        );

        plotPrediction2D.OutputDirectory = Path.Combine(plotPrediction2D.OutputDirectory, pointProcessModelDirectory);
        plotPrediction2D.Show<float>(prediction);
        plotPrediction2D.Save(png: true);

        Assert.IsTrue(true);
    }
}