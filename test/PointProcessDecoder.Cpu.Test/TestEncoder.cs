using PointProcessDecoder.Test.Common;
using PointProcessDecoder.Core;

namespace PointProcessDecoder.Cpu.Test;

[TestClass]
public class TestEncoder
{
    [TestMethod]
    public void TestSortedSpikeEncoderDensity1D()
    {
        EncoderUtilities.SortedSpikeEncoder(
            bandwidth: [5.0],
            numDimensions: 1,
            evaluationSteps: [50],
            steps: 200,
            cycles: 10,
            min: [0],
            max: [100],
            numNeurons: 40,
            placeFieldRadius: 8.0,
            firingThreshold: 0.2
        );
    }

    [TestMethod]
    public void TestSortedSpikeEncoderDensity2D()
    {
        EncoderUtilities.SortedSpikeEncoder(
            bandwidth: [5.0, 5.0],
            numDimensions: 2,
            evaluationSteps: [50, 50],
            steps: 200,
            cycles: 10,
            min: [0, 0],
            max: [100, 100],
            numNeurons: 40,
            placeFieldRadius: 8.0,
            firingThreshold: 0.2,
            modelDirectory: "SortedSpikeEncoder2D"
        );
    }

    [TestMethod]
    public void TestSortedSpikeEncoderCompression1D()
    {
        EncoderUtilities.SortedSpikeEncoder(
            estimationMethod: Core.Estimation.EstimationMethod.KernelCompression,
            distanceThreshold: 1.5,
            bandwidth: [5.0],
            numDimensions: 1,
            evaluationSteps: [50],
            steps: 200,
            cycles: 10,
            min: [0],
            max: [100],
            numNeurons: 40,
            placeFieldRadius: 8.0,
            firingThreshold: 0.2
        );
    }

    [TestMethod]
    public void TestSortedSpikeEncoderCompression2D()
    {
        EncoderUtilities.SortedSpikeEncoder(
            estimationMethod: Core.Estimation.EstimationMethod.KernelCompression,
            distanceThreshold: 1.5,
            bandwidth: [5.0, 5.0],
            numDimensions: 2,
            evaluationSteps: [50, 50],
            steps: 200,
            cycles: 10,
            min: [0, 0],
            max: [100, 100],
            numNeurons: 40,
            placeFieldRadius: 8.0,
            firingThreshold: 0.2,
            modelDirectory: "SortedSpikeEncoder2D"
        );
    }

    [TestMethod]
    public void TestClusterlessMarkEncoderDensity1D()
    {
        EncoderUtilities.ClusterlessMarkEncoder(
            observationBandwidth: [5.0],
            numDimensions: 1,
            evaluationSteps: [50],
            steps: 200,
            cycles: 10,
            min: [0],
            max: [100],
            markDimensions: 4,
            markChannels: 8,
            markBandwidth: [1.0, 1.0, 1.0, 1.0],
            numNeurons: 40,
            placeFieldRadius: 8.0,
            firingThreshold: 0.2
        );
    }

    [TestMethod]
    public void TestClusterlessMarkEncoderDensity2D()
    {
        EncoderUtilities.ClusterlessMarkEncoder(
            observationBandwidth: [5.0, 5.0],
            numDimensions: 2,
            evaluationSteps: [50, 50],
            steps: 200,
            cycles: 10,
            min: [0, 0],
            max: [100, 100],
            markDimensions: 4,
            markChannels: 8,
            markBandwidth: [1.0, 1.0, 1.0, 1.0],
            numNeurons: 40,
            placeFieldRadius: 8.0,
            firingThreshold: 0.2,
            modelDirectory: "ClusterlessMarkEncoder2D"
        );
    }
}