using static TorchSharp.torch;
using PointProcessDecoder.Core;
using PointProcessDecoder.Plot;
using PointProcessDecoder.Core.Estimation;
using PointProcessDecoder.Core.Transitions;
using PointProcessDecoder.Core.Encoder;
using PointProcessDecoder.Core.Decoder;

namespace PointProcessDecoder.Test;

[TestClass]
public class TestModel
{
    private int heatmapPadding = 10;
    private int seed = 0;
    private ScalarType scalarType = ScalarType.Float64;
    private Device device = CPU;
    private int testingSteps = 1800;
    private string outputDirectory = "TestModel";

    [TestMethod]
    public void TestPointProcessModelUniformDensity()
    {
        double[] bandwidth = [5];
        int numDimensions = 1;
        long evaluationSteps = 50;
        int steps = 200;
        int cycles = 10;
        double min = 0.0;
        double max = 100.0;
        int numNeurons = 40;
        double placeFieldRadius = 8.0;
        double firingThreshold = 0.2;

        var pointProcessModelDirectory = Path.Combine(outputDirectory, "PointProcessModelUniformDensity");
        var (position1D, spikingData) = Utilities.InitializeSimulation1D(
            steps: steps,
            cycles: cycles,
            min: min,
            max: max,
            numNeurons: numNeurons,
            placeFieldRadius: placeFieldRadius,
            firingThreshold: firingThreshold,
            scalarType: scalarType,
            seed: seed
        );

        var pointProcessModel = new PointProcessModel(
            EstimationMethod.KernelDensity,
            TransitionsType.Uniform,
            EncoderType.SortedSpikeEncoder,
            DecoderType.SortedSpikeDecoder,
            [min],
            [max],
            [evaluationSteps],
            bandwidth,
            latentDimensions: numDimensions,
            nUnits: numNeurons,
            device: device
        );

        pointProcessModel.Encode(position1D[TensorIndex.Slice(0, testingSteps)], spikingData[TensorIndex.Slice(0, testingSteps)]);
        var prediction = pointProcessModel.Decode(spikingData[TensorIndex.Slice(testingSteps)]);

        Heatmap plotPrediction = new(
            0,
            steps * cycles - testingSteps,
            min,
            max,
            title: "Prediction1D"
        );

        plotPrediction.OutputDirectory = Path.Combine(plotPrediction.OutputDirectory, pointProcessModelDirectory);
        plotPrediction.Show<float>(
            prediction, 
            concat([arange(steps * cycles - testingSteps).unsqueeze(-1), position1D[TensorIndex.Slice(testingSteps)]], dim: 1)
        );
        plotPrediction.Save(png: true);

        Assert.IsTrue(true);
    }

    [TestMethod]
    public void TestPointProcessModelRandomWalkDensity()
    {
        double[] bandwidth = [5];
        int numDimensions = 1;
        long evaluationSteps = 50;
        int steps = 200;
        int cycles = 10;
        double min = 0.0;
        double max = 100.0;
        int numNeurons = 40;
        double placeFieldRadius = 8.0;
        double firingThreshold = 0.2;
        double sigma = 1.0;

        var pointProcessModelDirectory = Path.Combine(outputDirectory, "PointProcessModelRandomWalkDensity");
        var (position1D, spikingData) = Utilities.InitializeSimulation1D(
            steps: steps,
            cycles: cycles,
            min: min,
            max: max,
            numNeurons: numNeurons,
            placeFieldRadius: placeFieldRadius,
            firingThreshold: firingThreshold,
            scalarType: scalarType,
            seed: seed
        );

        var pointProcessModel = new PointProcessModel(
            EstimationMethod.KernelDensity,
            TransitionsType.RandomWalk,
            EncoderType.SortedSpikeEncoder,
            DecoderType.SortedSpikeDecoder,
            [min],
            [max],
            [evaluationSteps],
            bandwidth,
            latentDimensions: numDimensions,
            nUnits: numNeurons,
            sigmaRandomWalk: [sigma],
            device: device
        );

        pointProcessModel.Encode(position1D[TensorIndex.Slice(0, testingSteps)], spikingData[TensorIndex.Slice(0, testingSteps)]);
        var prediction = pointProcessModel.Decode(spikingData[TensorIndex.Slice(testingSteps)]);

        Heatmap plotPrediction = new(
            0,
            steps * cycles - testingSteps,
            min,
            max,
            title: "Prediction1D"
        );

        plotPrediction.OutputDirectory = Path.Combine(plotPrediction.OutputDirectory, pointProcessModelDirectory);
        plotPrediction.Show<float>(
            prediction,
            concat([arange(steps * cycles - testingSteps).unsqueeze(-1), position1D[TensorIndex.Slice(testingSteps)]], dim: 1)
        );
        plotPrediction.Save(png: true);

        Assert.IsTrue(true);
    }

    [TestMethod]
    public void TestPointProcessModelRandomWalkCompression()
    {
        double[] bandwidth = [5];
        int numDimensions = 1;
        long evaluationSteps = 50;
        int steps = 200;
        int cycles = 10;
        double min = 0.0;
        double max = 100.0;
        int numNeurons = 40;
        double placeFieldRadius = 8.0;
        double firingThreshold = 0.2;
        double sigma = 1.0;
        double distanceThreshold = 1.5;

        var pointProcessModelDirectory = Path.Combine(outputDirectory, "PointProcessModelRandomWalkCompression");
        var (position1D, spikingData) = Utilities.InitializeSimulation1D(
            steps: steps,
            cycles: cycles,
            min: min,
            max: max,
            numNeurons: numNeurons,
            placeFieldRadius: placeFieldRadius,
            firingThreshold: firingThreshold,
            scalarType: scalarType,
            seed: seed
        );

        var pointProcessModel = new PointProcessModel(
            EstimationMethod.KernelCompression,
            TransitionsType.RandomWalk,
            EncoderType.SortedSpikeEncoder,
            DecoderType.SortedSpikeDecoder,
            [min],
            [max],
            [evaluationSteps],
            bandwidth,
            latentDimensions: numDimensions,
            nUnits: numNeurons,
            distanceThreshold: distanceThreshold,
            sigmaRandomWalk: [sigma],
            device: device
        );

        pointProcessModel.Encode(position1D[TensorIndex.Slice(0, testingSteps)], spikingData[TensorIndex.Slice(0, testingSteps)]);
        var prediction = pointProcessModel.Decode(spikingData[TensorIndex.Slice(testingSteps)]);

        Heatmap plotPrediction = new(
            0,
            steps * cycles - testingSteps,
            min,
            max,
            title: "Prediction1D"
        );

        plotPrediction.OutputDirectory = Path.Combine(plotPrediction.OutputDirectory, pointProcessModelDirectory);
        plotPrediction.Show<float>(
            prediction,
            concat([arange(steps * cycles - testingSteps).unsqueeze(-1), position1D[TensorIndex.Slice(testingSteps)]], dim: 1)
        );
        plotPrediction.Save(png: true);

        Assert.IsTrue(true);
    }

    [TestMethod]
    public void TestPointProcessModelUniformDensity2D()
    {
        double[] bandwidth = [5, 5];
        int numDimensions = 2;
        long[] evaluationSteps = [50, 50];
        int steps = 200;
        int cycles = 10;
        double xMin = 0.0;
        double xMax = 100.0;
        double yMin = 0.0;
        double yMax = 100.0;
        int numNeurons = 40;
        double placeFieldRadius = 8.0;
        double firingThreshold = 0.2;
        double scale = 0.1;

        var pointProcessModelDirectory = Path.Combine(outputDirectory, "PointProcessModelUniformDensity2D");
        var (position2D, spikingData) = Utilities.InitializeSimulation2D(
            steps: steps,
            cycles: cycles,
            xMin: xMin,
            xMax: xMax,
            yMin: yMin,
            yMax: yMax,
            numNeurons: numNeurons,
            placeFieldRadius: placeFieldRadius,
            firingThreshold: firingThreshold,
            scale: scale,
            scalarType: scalarType,
            seed: seed
        );

        var pointProcessModel = new PointProcessModel(
            EstimationMethod.KernelDensity,
            TransitionsType.Uniform,
            EncoderType.SortedSpikeEncoder,
            DecoderType.SortedSpikeDecoder,
            [xMin, yMin],
            [xMax, yMax],
            evaluationSteps,
            bandwidth,
            latentDimensions: numDimensions,
            nUnits: numNeurons,
            device: device
        );

        pointProcessModel.Encode(position2D[TensorIndex.Slice(0, testingSteps)], spikingData[TensorIndex.Slice(0, testingSteps)]);
        var prediction = pointProcessModel.Decode(spikingData[TensorIndex.Slice(testingSteps)]);
        prediction = prediction[0].reshape(evaluationSteps);

        Heatmap plotPrediction = new(
            xMin,
            xMax,
            yMin,
            yMax,
            title: "Prediction2D"
        );

        plotPrediction.OutputDirectory = Path.Combine(plotPrediction.OutputDirectory, pointProcessModelDirectory);
        plotPrediction.Show<float>(
            prediction,
            position2D[TensorIndex.Slice(testingSteps)]
        );
        plotPrediction.Save(png: true);

        Assert.IsTrue(true);
    }
}