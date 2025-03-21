using PointProcessDecoder.Core.Transitions;
using PointProcessDecoder.Core.Estimation;
using PointProcessDecoder.Test.Common;
using PointProcessDecoder.Core;
using static TorchSharp.torch;
using System.Drawing;

namespace PointProcessDecoder.Cpu.Test;

[TestClass]
public class TestModel
{
    [TestMethod]
    public void TestPointProcessModelSortedUnitsUniformDensitySimulatedData()
    {
        SortedUnitsUtilities.BayesianStateSpaceSortedUnitsSimulatedData(
            transitionsType: TransitionsType.Uniform,
            bandwidth: [2],
            firingThreshold: 0.2,
            modelDirectory: "SimulatedData"
        );
    }

    [TestMethod]
    public void TestPointProcessModelSortedUnitsRandomWalkDensitySimulatedData()
    {
        SortedUnitsUtilities.BayesianStateSpaceSortedUnitsSimulatedData(
            transitionsType: TransitionsType.RandomWalk,
            sigma: 0.5,
            modelDirectory: "SimulatedData1D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelSortedUnitsRandomWalkCompressionSimulatedData()
    {
        SortedUnitsUtilities.BayesianStateSpaceSortedUnitsSimulatedData(
            transitionsType: TransitionsType.RandomWalk,
            estimationMethod: EstimationMethod.KernelCompression,
            sigma: 0.5,
            distanceThreshold: 1.5,
            modelDirectory: "SimulatedData1D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelSortedUnitsUniformDensitySimulatedData2D()
    {
        SortedUnitsUtilities.BayesianStateSpaceSortedUnitsSimulatedData(
            bandwidth: [5.0, 5.0],
            dimensions: 2,
            evaluationSteps: [50, 50],
            min: [0, 0],
            max: [100, 100],
            scale: 0.1,
            modelDirectory: "SimulatedData2D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelSortedUnitsRandomWalkDensitySimulatedData2D()
    {
        SortedUnitsUtilities.BayesianStateSpaceSortedUnitsSimulatedData(
            bandwidth: [5.0, 5.0],
            dimensions: 2,
            evaluationSteps: [50, 50],
            min: [0, 0],
            max: [100, 100],
            scale: 0.1,
            transitionsType: TransitionsType.RandomWalk,
            sigma: 0.1,
            modelDirectory: "SimulatedData2D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelSortedUnitsRandomWalkCompressionSimulatedData2D()
    {
        SortedUnitsUtilities.BayesianStateSpaceSortedUnitsSimulatedData(
            bandwidth: [5.0, 5.0],
            dimensions: 2,
            evaluationSteps: [50, 50],
            min: [0, 0],
            max: [100, 100],
            scale: 0.1,
            transitionsType: TransitionsType.RandomWalk,
            estimationMethod: EstimationMethod.KernelCompression,
            sigma: 0.1,
            distanceThreshold: 1.5,
            modelDirectory: "SimulatedData2D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelSortedUnitsUniformDensityRealData2D()
    {
        SortedUnitsUtilities.BayesianStateSpaceSortedUnitsRealData(
            bandwidth: [0.5, 0.5],
            dimensions: 2,
            evaluationSteps: [50, 50],
            minVals: [0, 0],
            maxVals: [120, 120],
            testFraction: 0.01,
            trainingFraction: 0.2,
            modelDirectory: "RealData2D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelSortedUnitsRandomWalkCompressionRealData2D()
    {
        SortedUnitsUtilities.BayesianStateSpaceSortedUnitsRealData(
            bandwidth: [0.5, 0.5],
            dimensions: 2,
            evaluationSteps: [50, 50],
            minVals: [0, 0],
            maxVals: [120, 120],
            sigma: 0.25,
            distanceThreshold: 1.5,
            testFraction: 0.01,
            trainingFraction: 0.2,
            modelDirectory: "RealData2D",
            transitionsType: TransitionsType.RandomWalk,
            estimationMethod: EstimationMethod.KernelCompression
        );
    }

    [TestMethod]
    public void TestPointProcessModelClusterlessMarksUniformDensitySimulatedData()
    {
        ClusterlessMarksUtilities.BayesianStateSpaceClusterlessMarksSimulated(
            transitionsType: TransitionsType.Uniform,
            observationBandwidth: [2],
            markBandwidth: [1,1,1,1],
            firingThreshold: 0.2,
            noiseScale: 0.5,
            modelDirectory: "SimulatedData"
        );
    }

    [TestMethod]
    public void TestPointProcessModelClusterlessMarksUniformCompressionSimulatedData()
    {
        ClusterlessMarksUtilities.BayesianStateSpaceClusterlessMarksSimulated(
            transitionsType: TransitionsType.Uniform,
            estimationMethod: EstimationMethod.KernelCompression,
            distanceThreshold: 1.5,
            observationBandwidth: [2],
            markBandwidth: [1,1,1,1],
            firingThreshold: 0.2,
            noiseScale: 1.0,
            modelDirectory: "SimulatedData"
        );
    }

    [TestMethod]
    public void TestPointProcessModelClusterlessMarksUniformDensitySimulatedData2D()
    {
        ClusterlessMarksUtilities.BayesianStateSpaceClusterlessMarksSimulated(
            transitionsType: TransitionsType.Uniform,
            observationBandwidth: [5,5],
            dimensions : 2,
            evaluationSteps: [50, 50],
            min: [0, 0],
            max: [100, 100],
            scale: 0.1,
            markBandwidth: [0.5,0.5,0.5,0.5],
            firingThreshold: 0.4,
            noiseScale: 0.5,
            modelDirectory: "SimulatedData2D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelClusterlessMarksUniformCompressionSimulatedData2D()
    {
        ClusterlessMarksUtilities.BayesianStateSpaceClusterlessMarksSimulated(
            transitionsType: TransitionsType.Uniform,
            estimationMethod: EstimationMethod.KernelCompression,
            distanceThreshold: 1.5,
            observationBandwidth: [5,5],
            dimensions : 2,
            evaluationSteps: [50, 50],
            min: [0, 0],
            max: [100, 100],
            scale: 0.1,
            markBandwidth: [0.5,0.5,0.5,0.5],
            firingThreshold: 0.4,
            noiseScale: 0.5,
            modelDirectory: "SimulatedData2D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelClusterlessMarksRandomWalkDensitySimulatedData()
    {
        ClusterlessMarksUtilities.BayesianStateSpaceClusterlessMarksSimulated(
            transitionsType: TransitionsType.RandomWalk,
            observationBandwidth: [5],
            sigma: 3,
            markBandwidth: [2,2,2,2],
            firingThreshold: 0.5,
            noiseScale: 0.5,
            modelDirectory: "SimulatedData"
        );
    }

    [TestMethod]
    public void TestPointProcessModelClusterlessMarksRandomWalkCompressionSimulatedData()
    {
        ClusterlessMarksUtilities.BayesianStateSpaceClusterlessMarksSimulated(
            transitionsType: TransitionsType.RandomWalk,
            estimationMethod: EstimationMethod.KernelCompression,
            distanceThreshold: 1.5,
            observationBandwidth: [5],
            sigma: 3,
            markBandwidth: [2,2,2,2],
            firingThreshold: 0.5,
            noiseScale: 0.5,
            modelDirectory: "SimulatedData"
        );
    }

    [TestMethod]
    public void TestPointProcessModelClusterlessMarksRandomWalkDensitySimulatedData2D()
    {
        ClusterlessMarksUtilities.BayesianStateSpaceClusterlessMarksSimulated(
            transitionsType: TransitionsType.RandomWalk,
            observationBandwidth: [5, 5],
            sigma: 1,
            dimensions : 2,
            evaluationSteps: [50, 50],
            min: [0, 0],
            max: [100, 100],
            scale: 0.1,
            markBandwidth: [0.5,0.5,0.5,0.5],
            firingThreshold: 0.4,
            noiseScale: 0.5,
            modelDirectory: "SimulatedData2D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelClusterlessMarksRandomWalkCompressionSimulatedData2D()
    {
        ClusterlessMarksUtilities.BayesianStateSpaceClusterlessMarksSimulated(
            transitionsType: TransitionsType.RandomWalk,
            estimationMethod: EstimationMethod.KernelCompression,
            distanceThreshold: 1,
            observationBandwidth: [10, 10],
            sigma: 10,
            dimensions : 2,
            evaluationSteps: [50, 50],
            min: [0, 0],
            max: [100, 100],
            scale: 0.1,
            markBandwidth: [2,2,2,2],
            firingThreshold: 0.4,
            noiseScale: 0.5,
            modelDirectory: "SimulatedData2D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelClusterlessMarksUniformDensityRealData2D()
    {
        ClusterlessMarksUtilities.BayesianStateSpaceClusterlessMarksRealData(
            observationBandwidth: [0.5, 0.5],
            dimensions: 2,
            evaluationSteps: [50, 50],
            minVals: [0, 0],
            maxVals: [120, 120],
            trainingFraction: 0.8,
            modelDirectory: "RealData2D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelClusterlessMarksRandomWalkCompressionRealData2D()
    {
        ClusterlessMarksUtilities.BayesianStateSpaceClusterlessMarksRealData(
            transitionsType: TransitionsType.RandomWalk,
            estimationMethod: EstimationMethod.KernelCompression,
            distanceThreshold: 1.5,
            sigma: 0.25,
            observationBandwidth: [0.5, 0.5],
            dimensions: 2,
            evaluationSteps: [50, 50],
            minVals: [0, 0],
            maxVals: [120, 120],
            trainingFraction: 0.8,
            modelDirectory: "RealData2D"
        );
    }

    Tensor ReadBinaryFile(
        string binary_file
    )
    {
        byte[] fileBytes = File.ReadAllBytes(binary_file);
        int elementCount = fileBytes.Length / sizeof(double);
        double[] doubleArray = new double[elementCount];
        Buffer.BlockCopy(fileBytes, 0, doubleArray, 0, fileBytes.Length);
        Tensor t = tensor(doubleArray);
        return t;
    }

    (Tensor, Tensor) InitializeRealData(
        string positionFile,
        string marksFile
    )
    {
        var position = ReadBinaryFile(positionFile);
        var marks = ReadBinaryFile(marksFile);
        return (position, marks);
    }

    [TestMethod]
    public void TestPointProcessModelClusterlessMarksRandomWalkCompressionRealData2DBatchedProcessing()
    {
        string positionFile = "../../../../data/position.bin";
        string marksFile = "../../../../data/marks.bin";

        int markDimensions = 4;
        int markChannels = 28;

        var (position, marks) = InitializeRealData(
            positionFile: positionFile,
            marksFile: marksFile
        );

        position = position.reshape(-1, 2);
        marks = marks.reshape(position.shape[0], markDimensions, markChannels);

        var pointProcessModel = new PointProcessModel(
            estimationMethod: EstimationMethod.KernelCompression,
            transitionsType: TransitionsType.RandomWalk,
            encoderType: Core.Encoder.EncoderType.ClusterlessMarkEncoder,
            decoderType: Core.Decoder.DecoderType.StateSpaceDecoder,
            stateSpaceType: Core.StateSpace.StateSpaceType.DiscreteUniformStateSpace,
            likelihoodType: Core.Likelihood.LikelihoodType.Clusterless,
            minStateSpace: [0, 0],
            maxStateSpace: [120, 120],
            stepsStateSpace: [50, 50],
            observationBandwidth: [2, 2],
            stateSpaceDimensions: 2,
            markDimensions: markDimensions,
            markChannels: markChannels,
            markBandwidth: [1, 1, 1, 1],
            distanceThreshold: 1.5,
            sigmaRandomWalk: 5
        );

        int batchSize = 60;

        for (int i = 0; i < 10; i++) {
            pointProcessModel.Encode(
                position[TensorIndex.Slice(i * batchSize, (i + 1) * batchSize)],
                marks[TensorIndex.Slice(i * batchSize, (i + 1) * batchSize)]
            );
        }

        var prediction = pointProcessModel.Decode(marks[TensorIndex.Slice(10 * batchSize, 11 * batchSize)])
            .sum(dim: 0);

        Assert.IsFalse(prediction.isnan().any().item<bool>());
    }

    [TestMethod]
    public void CompareClusterlessEncodingBatchSizes()
    {
        string positionFile = "../../../../data/position.bin";
        string marksFile = "../../../../data/marks.bin";

        int markDimensions = 4;
        int markChannels = 28;

        var (position, marks) = InitializeRealData(
            positionFile: positionFile,
            marksFile: marksFile
        );

        position = position.reshape(-1, 2);
        marks = marks.reshape(position.shape[0], markDimensions, markChannels);

        var pointProcessModel = new PointProcessModel(
            estimationMethod: EstimationMethod.KernelCompression,
            transitionsType: TransitionsType.RandomWalk,
            encoderType: Core.Encoder.EncoderType.ClusterlessMarkEncoder,
            decoderType: Core.Decoder.DecoderType.StateSpaceDecoder,
            stateSpaceType: Core.StateSpace.StateSpaceType.DiscreteUniformStateSpace,
            likelihoodType: Core.Likelihood.LikelihoodType.Clusterless,
            minStateSpace: [0, 0],
            maxStateSpace: [120, 120],
            stepsStateSpace: [50, 50],
            observationBandwidth: [5, 5],
            stateSpaceDimensions: 2,
            markDimensions: markDimensions,
            markChannels: markChannels,
            markBandwidth: [1, 1, 1, 1],
            distanceThreshold: 1.5,
            sigmaRandomWalk: 1
        );

        // Encode in batches

        int nBatches = 100;
        int batchSize = 60;

        for (int i = 0; i < nBatches; i++) {
            pointProcessModel.Encode(
                position[TensorIndex.Slice(i * batchSize, (i + 1) * batchSize)],
                marks[TensorIndex.Slice(i * batchSize, (i + 1) * batchSize)]
            );
        }

        var prediction1 = pointProcessModel.Decode(marks[TensorIndex.Slice(nBatches * batchSize, (nBatches + 1) * batchSize)])
            .sum(dim: 0);

        pointProcessModel = new PointProcessModel(
            estimationMethod: EstimationMethod.KernelCompression,
            transitionsType: TransitionsType.RandomWalk,
            encoderType: Core.Encoder.EncoderType.ClusterlessMarkEncoder,
            decoderType: Core.Decoder.DecoderType.StateSpaceDecoder,
            stateSpaceType: Core.StateSpace.StateSpaceType.DiscreteUniformStateSpace,
            likelihoodType: Core.Likelihood.LikelihoodType.Clusterless,
            minStateSpace: [0, 0],
            maxStateSpace: [120, 120],
            stepsStateSpace: [50, 50],
            observationBandwidth: [5, 5],
            stateSpaceDimensions: 2,
            markDimensions: markDimensions,
            markChannels: markChannels,
            markBandwidth: [1, 1, 1, 1],
            distanceThreshold: 1.5,
            sigmaRandomWalk: 1
        );

        // Encode all at once

        pointProcessModel.Encode(
            position[TensorIndex.Slice(0, nBatches * batchSize)],
            marks[TensorIndex.Slice(0, nBatches * batchSize)]
        );

        var prediction2 = pointProcessModel.Decode(marks[TensorIndex.Slice(nBatches * batchSize, (nBatches + 1) * batchSize)])
            .sum(dim: 0);

        Assert.AreEqual(prediction1, prediction2);
    }

    [TestMethod]
    public void CompareSortedUnitsEncodingBatchSizes()
    {
        string positionFile = "../../../../data/position.bin";
        string spikesFile = "../../../../data/spike_counts.bin";

        var (position, spikingData) = InitializeRealData(
            positionFile: positionFile,
            marksFile: spikesFile
        );

        position = position.reshape(-1, 2);
        spikingData = spikingData.reshape(position.shape[0], -1)
            .to_type(ScalarType.Int32);
        var numNeurons = (int)spikingData.shape[1];

        var pointProcessModel = new PointProcessModel(
            estimationMethod: EstimationMethod.KernelCompression,
            transitionsType: TransitionsType.RandomWalk,
            encoderType: Core.Encoder.EncoderType.SortedSpikeEncoder,
            decoderType: Core.Decoder.DecoderType.StateSpaceDecoder,
            stateSpaceType: Core.StateSpace.StateSpaceType.DiscreteUniformStateSpace,
            likelihoodType: Core.Likelihood.LikelihoodType.Poisson,
            minStateSpace: [0, 0],
            maxStateSpace: [120, 120],
            stepsStateSpace: [50, 50],
            observationBandwidth: [5, 5],
            stateSpaceDimensions: 2,
            distanceThreshold: 1.5,
            nUnits: numNeurons,
            sigmaRandomWalk: 1
        );

        // Encode in batches

        int nBatches = 100;
        int batchSize = 60;

        for (int i = 0; i < nBatches; i++) {
            pointProcessModel.Encode(
                position[TensorIndex.Slice(i * batchSize, (i + 1) * batchSize)],
                spikingData[TensorIndex.Slice(i * batchSize, (i + 1) * batchSize)]
            );
        }

        var prediction1 = pointProcessModel.Decode(spikingData[TensorIndex.Slice(nBatches * batchSize, (nBatches + 1) * batchSize)])
            .sum(dim: 0);

        pointProcessModel = new PointProcessModel(
            estimationMethod: EstimationMethod.KernelCompression,
            transitionsType: TransitionsType.RandomWalk,
            encoderType: Core.Encoder.EncoderType.SortedSpikeEncoder,
            decoderType: Core.Decoder.DecoderType.StateSpaceDecoder,
            stateSpaceType: Core.StateSpace.StateSpaceType.DiscreteUniformStateSpace,
            likelihoodType: Core.Likelihood.LikelihoodType.Poisson,
            minStateSpace: [0, 0],
            maxStateSpace: [120, 120],
            stepsStateSpace: [50, 50],
            observationBandwidth: [5, 5],
            stateSpaceDimensions: 2,
            distanceThreshold: 1.5,
            nUnits: numNeurons,
            sigmaRandomWalk: 1
        );

        // Encode all at once

        pointProcessModel.Encode(
            position[TensorIndex.Slice(0, nBatches * batchSize)],
            spikingData[TensorIndex.Slice(0, nBatches * batchSize)]
        );

        var prediction2 = pointProcessModel.Decode(spikingData[TensorIndex.Slice(nBatches * batchSize, (nBatches + 1) * batchSize)])
            .sum(dim: 0);

        Assert.AreEqual(prediction1, prediction2);
    }

    [TestMethod]
    public void TestKernelLimit()
    {
        string positionFile = "../../../../data/position.bin";
        string marksFile = "../../../../data/marks.bin";

        int markDimensions = 4;
        int markChannels = 28;

        var (position, marks) = InitializeRealData(
            positionFile: positionFile,
            marksFile: marksFile
        );

        position = position.reshape(-1, 2);
        marks = marks.reshape(position.shape[0], markDimensions, markChannels);

        var pointProcessModel = new PointProcessModel(
            estimationMethod: EstimationMethod.KernelCompression,
            transitionsType: TransitionsType.RandomWalk,
            encoderType: Core.Encoder.EncoderType.ClusterlessMarkEncoder,
            decoderType: Core.Decoder.DecoderType.StateSpaceDecoder,
            stateSpaceType: Core.StateSpace.StateSpaceType.DiscreteUniformStateSpace,
            likelihoodType: Core.Likelihood.LikelihoodType.Clusterless,
            minStateSpace: [0, 0],
            maxStateSpace: [120, 120],
            stepsStateSpace: [50, 50],
            observationBandwidth: [2, 2],
            stateSpaceDimensions: 2,
            markDimensions: markDimensions,
            markChannels: markChannels,
            markBandwidth: [1, 1, 1, 1],
            distanceThreshold: 1.5,
            sigmaRandomWalk: 5
        );

        int nBatches = 10;
        int batchSize = 1000;

        for (int i = 0; i < nBatches; i++) {
            pointProcessModel.Encode(
                position[TensorIndex.Slice(i * batchSize, (i + 1) * batchSize)],
                marks[TensorIndex.Slice(i * batchSize, (i + 1) * batchSize)]
            );
        }

        var kernelCounts = pointProcessModel.Encoder.Estimations.Select(e => e.Kernels.shape[0]).ToList();

        Assert.IsTrue(kernelCounts.Any(k => k > 100));

        var pointProcessModelLimited = new PointProcessModel(
            estimationMethod: EstimationMethod.KernelCompression,
            transitionsType: TransitionsType.RandomWalk,
            encoderType: Core.Encoder.EncoderType.ClusterlessMarkEncoder,
            decoderType: Core.Decoder.DecoderType.StateSpaceDecoder,
            stateSpaceType: Core.StateSpace.StateSpaceType.DiscreteUniformStateSpace,
            likelihoodType: Core.Likelihood.LikelihoodType.Clusterless,
            minStateSpace: [0, 0],
            maxStateSpace: [120, 120],
            stepsStateSpace: [50, 50],
            observationBandwidth: [2, 2],
            stateSpaceDimensions: 2,
            markDimensions: markDimensions,
            markChannels: markChannels,
            markBandwidth: [1, 1, 1, 1],
            distanceThreshold: 1.5,
            sigmaRandomWalk: 5,
            kernelLimit: 100
        );

        for (int i = 0; i < nBatches; i++) {
            pointProcessModelLimited.Encode(
                position[TensorIndex.Slice(i * batchSize, (i + 1) * batchSize)],
                marks[TensorIndex.Slice(i * batchSize, (i + 1) * batchSize)]
            );
        }

        var kernelCountsLimited = pointProcessModelLimited.Encoder.Estimations.Select(e => e.Kernels.shape[0]).ToList();

        Assert.IsTrue(kernelCountsLimited.All(k => k <= 100));
    }
}