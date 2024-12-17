using static TorchSharp.torch;
using PointProcessDecoder.Core;
using PointProcessDecoder.Plot;
using PointProcessDecoder.Core.Estimation;
using PointProcessDecoder.Core.Transitions;
using PointProcessDecoder.Core.Encoder;
using PointProcessDecoder.Core.Decoder;
using static PointProcessDecoder.Test.Common.Utilities;

namespace PointProcessDecoder.Cuda.Test;

[TestClass]
public class TestModel
{
    private int seed = 0;
    private ScalarType scalarType = ScalarType.Float32;
    private Device device = CUDA;
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
        int nTraining = 1800;
        int nTesting = 200;

        var pointProcessModelDirectory = Path.Combine(outputDirectory, "PointProcessModelUniformDensity");
        var (position1D, spikingData) = InitializeSimulation1D(
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

        pointProcessModel.Encode(position1D[TensorIndex.Slice(0, nTraining)], spikingData[TensorIndex.Slice(0, nTraining)]);
        var prediction = pointProcessModel.Decode(spikingData[TensorIndex.Slice(nTraining, nTraining + nTesting)]);

        Heatmap plotPrediction = new(
            0,
            steps * cycles - nTraining,
            min,
            max,
            title: "Prediction1D"
        );

        plotPrediction.OutputDirectory = Path.Combine(plotPrediction.OutputDirectory, pointProcessModelDirectory);
        plotPrediction.Show<float>(
            prediction, 
            concat([arange(steps * cycles - nTraining).unsqueeze(-1), position1D[TensorIndex.Slice(nTraining, nTraining + nTesting)]], dim: 1)
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
        int nTraining = 1800;
        int nTesting = 200;

        var pointProcessModelDirectory = Path.Combine(outputDirectory, "PointProcessModelRandomWalkDensity");
        var (position1D, spikingData) = InitializeSimulation1D(
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

        pointProcessModel.Encode(position1D[TensorIndex.Slice(0, nTraining)], spikingData[TensorIndex.Slice(0, nTraining)]);
        var prediction = pointProcessModel.Decode(spikingData[TensorIndex.Slice(nTraining, nTraining + nTesting)]);

        Heatmap plotPrediction = new(
            0,
            steps * cycles - nTraining,
            min,
            max,
            title: "Prediction1D"
        );

        plotPrediction.OutputDirectory = Path.Combine(plotPrediction.OutputDirectory, pointProcessModelDirectory);
        plotPrediction.Show<float>(
            prediction,
            concat([arange(steps * cycles - nTraining).unsqueeze(-1), position1D[TensorIndex.Slice(nTraining, nTraining + nTesting)]], dim: 1)
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
        int nTraining = 1800;
        int nTesting = 200;

        var pointProcessModelDirectory = Path.Combine(outputDirectory, "PointProcessModelRandomWalkCompression");
        var (position1D, spikingData) = InitializeSimulation1D(
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

        pointProcessModel.Encode(position1D[TensorIndex.Slice(0, nTraining)], spikingData[TensorIndex.Slice(0, nTraining)]);
        var prediction = pointProcessModel.Decode(spikingData[TensorIndex.Slice(nTraining, nTraining + nTesting)]);

        Heatmap plotPrediction = new(
            0,
            steps * cycles - nTraining,
            min,
            max,
            title: "Prediction1D"
        );

        plotPrediction.OutputDirectory = Path.Combine(plotPrediction.OutputDirectory, pointProcessModelDirectory);
        plotPrediction.Show<float>(
            prediction,
            concat([arange(steps * cycles - nTraining).unsqueeze(-1), position1D[TensorIndex.Slice(nTraining, nTraining + nTesting)]], dim: 1)
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
        int nTraining = 1800;
        int nTesting = 200;

        var pointProcessModelDirectory = Path.Combine(outputDirectory, "PointProcessModelUniformDensity2D");
        var (position2D, spikingData) = InitializeSimulation2D(
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

        pointProcessModel.Encode(position2D[TensorIndex.Slice(0, nTraining)], spikingData[TensorIndex.Slice(0, nTraining)]);
        var prediction = pointProcessModel.Decode(spikingData[TensorIndex.Slice(nTraining, nTraining + nTesting)]);
        prediction = (prediction.sum(dim: 0) / prediction.sum()).reshape(evaluationSteps);

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
            position2D[TensorIndex.Slice(nTraining, nTraining + nTesting)]
        );
        plotPrediction.Save(png: true);

        Assert.IsTrue(true);
    }

    [TestMethod]
    public void TestPointProcessModelRandomWalkDensity2D()
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
        double[] sigma = [100, 100];
        int nTraining = 1800;
        int nTesting = 200;

        var pointProcessModelDirectory = Path.Combine(outputDirectory, "PointProcessModelRandomWalkDensity2D");
        var (position2D, spikingData) = InitializeSimulation2D(
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
            TransitionsType.RandomWalk,
            EncoderType.SortedSpikeEncoder,
            DecoderType.SortedSpikeDecoder,
            [xMin, yMin],
            [xMax, yMax],
            evaluationSteps,
            bandwidth,
            latentDimensions: numDimensions,
            nUnits: numNeurons,
            device: device,
            sigmaRandomWalk: sigma
        );

        pointProcessModel.Encode(position2D[TensorIndex.Slice(0, nTraining)], spikingData[TensorIndex.Slice(0, nTraining)]);
        var prediction = pointProcessModel.Decode(spikingData[TensorIndex.Slice(nTraining, nTraining + nTesting)]);
        prediction = (prediction.sum(dim: 0) / prediction.sum()).reshape(evaluationSteps);

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
            position2D[TensorIndex.Slice(nTraining, nTraining + nTesting)]
        );
        plotPrediction.Save(png: true);

        Assert.IsTrue(true);
    }

    [TestMethod]
    public void TestPointProcessModelRandomWalkCompression2D()
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
        double distanceThreshold = 1.5;
        double[] sigma = [100, 100];
        int nTraining = 1800;
        int nTesting = 200;

        var pointProcessModelDirectory = Path.Combine(outputDirectory, "PointProcessModelRandomWalkCompression2D");
        var (position2D, spikingData) = InitializeSimulation2D(
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
            EstimationMethod.KernelCompression,
            TransitionsType.RandomWalk,
            EncoderType.SortedSpikeEncoder,
            DecoderType.SortedSpikeDecoder,
            [xMin, yMin],
            [xMax, yMax],
            evaluationSteps,
            bandwidth,
            latentDimensions: numDimensions,
            nUnits: numNeurons,
            distanceThreshold: distanceThreshold,
            device: device,
            sigmaRandomWalk: sigma
        );

        pointProcessModel.Encode(position2D[TensorIndex.Slice(0, nTraining)], spikingData[TensorIndex.Slice(0, nTraining)]);
        var prediction = pointProcessModel.Decode(spikingData[TensorIndex.Slice(nTraining, nTraining + nTesting)]);
        prediction = (prediction.sum(dim: 0) / prediction.sum()).reshape(evaluationSteps);

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
            position2D[TensorIndex.Slice(nTraining, nTraining + nTesting)]
        );
        plotPrediction.Save(png: true);

        Assert.IsTrue(true);
    }

    [TestMethod]
    public void TestPointProcessModelRandomWalkDensityRealData2D()
    {
        double[] bandwidth = [5, 5];
        int numDimensions = 2;
        long[] evaluationSteps = [50, 50];
        double xMin = 0.0;
        double xMax = 120.0;
        double yMin = 0.0;
        double yMax = 120.0;
        double[] sigma = [1, 1];
        int nTraining = 180000;
        int nTesting = 20000;
        int batchSize = 1000;

        string positionFile = "../../../../data/positions_2D.bin";
        string spikesFile = "../../../../data/spike_times.bin";

        var pointProcessModelDirectory = Path.Combine(outputDirectory, "PointProcessModelRandomWalkDensityRealData2D");
        var (position, spikingData) = InitializeRealData(
            positionFile: positionFile,
            spikesFile: spikesFile,
            device: device,
            scalarType: scalarType
        );

        var position2D = position.reshape(-1, 2);
        spikingData = spikingData.reshape(position2D.shape[0], -1)
            .to_type(ScalarType.Bool);
        var numNeurons = (int)spikingData.shape[1];

        var pointProcessModel = new PointProcessModel(
            EstimationMethod.KernelDensity,
            TransitionsType.RandomWalk,
            EncoderType.SortedSpikeEncoder,
            DecoderType.SortedSpikeDecoder,
            [xMin, yMin],
            [xMax, yMax],
            evaluationSteps,
            bandwidth,
            latentDimensions: numDimensions,
            nUnits: numNeurons,
            device: device,
            sigmaRandomWalk: sigma
        );

        for (int i = 0; i < nTraining; i += batchSize)
        {
            var end = Math.Min(i + batchSize, nTraining);
            pointProcessModel.Encode(
                position2D[TensorIndex.Slice(i, end)],
                spikingData[TensorIndex.Slice(i, end)]
            );
        }

        for (int i = nTraining; i < nTraining + nTesting; i += batchSize)
        {
            var end = Math.Min(i + batchSize, nTraining + nTesting);
            var prediction = pointProcessModel.Decode(spikingData[TensorIndex.Slice(i, end)]);
            prediction = (prediction.sum(dim: 0) / prediction.sum()).reshape(evaluationSteps);

            Heatmap plotPrediction = new(
                xMin,
                xMax,
                yMin,
                yMax,
                title: $"Prediction2D_{i}-{end}"
            );

            plotPrediction.OutputDirectory = Path.Combine(plotPrediction.OutputDirectory, pointProcessModelDirectory);
            plotPrediction.Show<float>(
                prediction,
                position2D[TensorIndex.Slice(i, end)]
            );
            plotPrediction.Save(png: true);
        }

        Assert.IsTrue(true);
    }

    [TestMethod]
    public void TestPointProcessModelRandomWalkCompressionRealData2D()
    {
        double[] bandwidth = [2, 2];
        int numDimensions = 2;
        long[] evaluationSteps = [50, 50];
        double xMin = 0.0;
        double xMax = 120.0;
        double yMin = 0.0;
        double yMax = 120.0;
        double[] sigma = [1, 1];
        int nTraining = 180000;
        int nTesting = 20000;
        int batchSize = 1000;
        double distanceThreshold = 1.5;

        string positionFile = "../../../../data/positions_2D.bin";
        string spikesFile = "../../../../data/spike_times.bin";

        var pointProcessModelDirectory = Path.Combine(outputDirectory, "PointProcessModelRandomWalkCompressionRealData2D");
        var (position, spikingData) = InitializeRealData(
            positionFile: positionFile,
            spikesFile: spikesFile,
            device: device,
            scalarType: scalarType
        );

        var position2D = position.reshape(-1, 2);
        spikingData = spikingData.reshape(position2D.shape[0], -1)
            .to_type(ScalarType.Bool);
        var numNeurons = (int)spikingData.shape[1];

        var pointProcessModel = new PointProcessModel(
            EstimationMethod.KernelCompression,
            TransitionsType.RandomWalk,
            EncoderType.SortedSpikeEncoder,
            DecoderType.SortedSpikeDecoder,
            [xMin, yMin],
            [xMax, yMax],
            evaluationSteps,
            bandwidth,
            latentDimensions: numDimensions,
            nUnits: numNeurons,
            device: device,
            distanceThreshold: distanceThreshold,
            sigmaRandomWalk: sigma
        );

        for (int i = 0; i < nTraining; i += batchSize)
        {
            var end = Math.Min(i + batchSize, nTraining);
            pointProcessModel.Encode(
                position2D[TensorIndex.Slice(i, end)],
                spikingData[TensorIndex.Slice(i, end)]
            );
        }

        for (int i = nTraining; i < nTraining + nTesting; i += batchSize)
        {
            var end = Math.Min(i + batchSize, nTraining + nTesting);
            var prediction = pointProcessModel.Decode(spikingData[TensorIndex.Slice(i, end)]);
            prediction = (prediction.sum(dim: 0) / prediction.sum()).reshape(evaluationSteps);

            Heatmap plotPrediction = new(
                xMin,
                xMax,
                yMin,
                yMax,
                title: $"Prediction2D_{i}-{end}"
            );

            plotPrediction.OutputDirectory = Path.Combine(plotPrediction.OutputDirectory, pointProcessModelDirectory);
            plotPrediction.Show<float>(
                prediction,
                position2D[TensorIndex.Slice(i, end)]
            );
            plotPrediction.Save(png: true);
        }

        Assert.IsTrue(true);
    }
}