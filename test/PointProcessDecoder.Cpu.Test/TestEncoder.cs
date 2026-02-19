using PointProcessDecoder.Test.Common;
using PointProcessDecoder.Core;
using static TorchSharp.torch;
using TorchSharp;

namespace PointProcessDecoder.Cpu.Test;

[TestClass]
public class TestEncoder
{
    [TestMethod]
    public void TestSortedSpikeEncoderDensity1D()
    {
        EncoderUtilities.SortedSpikeEncoder(
            covariateBandwidth: [5.0],
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
            covariateBandwidth: [5.0, 5.0],
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
            covariateBandwidth: [5.0],
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
            covariateBandwidth: [5.0, 5.0],
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
            covariateBandwidth: [5.0],
            numDimensions: 1,
            evaluationSteps: [50],
            steps: 200,
            cycles: 10,
            min: [0],
            max: [100],
            markDimensions: 4,
            numChannels: 8,
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
            covariateBandwidth: [5.0, 5.0],
            numDimensions: 2,
            evaluationSteps: [50, 50],
            steps: 200,
            cycles: 10,
            min: [0, 0],
            max: [100, 100],
            markDimensions: 4,
            numChannels: 8,
            markBandwidth: [1.0, 1.0, 1.0, 1.0],
            numNeurons: 40,
            placeFieldRadius: 8.0,
            firingThreshold: 0.2,
            modelDirectory: "ClusterlessMarkEncoder2D"
        );
    }

    [TestMethod]
    public void TestSortedSpikeEncoderDensity2DCovariateDownsampled()
    {
        var estimationMethod = Core.Estimation.EstimationMethod.KernelDensity;
        double[] covariateBandwidth = [5.0, 5.0];
        var numDimensions = 2;
        long[] evaluationSteps = [50, 50];
        var steps = 200;
        var cycles = 10;
        double[] min = [0, 0];
        double[] max = [100, 100];
        var numNeurons = 40;
        var placeFieldRadius = 8.0;
        var firingThreshold = 0.2;
        var scalarType = ScalarType.Float32;
        Device device = CPU;

        var stateSpace = new Core.StateSpace.DiscreteUniform(
            numDimensions,
            min,
            max,
            evaluationSteps,
            device: device,
            scalarType: scalarType
        );

        var (position2D, spikingData) = Simulation.Utilities.InitializeSimulation2D(
            steps,
            cycles,
            min[0],
            max[0],
            min[1],
            max[1],
            numNeurons,
            placeFieldRadius,
            firingThreshold
        );

        var sortedSpikeEncoder = new Core.Encoder.SortedSpikes(
            estimationMethod, 
            covariateBandwidth,
            numNeurons,
            stateSpace,
            device: device,
            scalarType: scalarType
        );

        var sortedSpikeEncoder2 = new Core.Encoder.SortedSpikes(
            estimationMethod, 
            covariateBandwidth,
            numNeurons,
            stateSpace,
            device: device,
            scalarType: scalarType
        );

        var numSamples = position2D.size(0);
        var sampleIndices = arange(0, numSamples, 100, dtype: int64);
        var numBatches = sampleIndices.size(0);
        var covariates = position2D.index_select(0, sampleIndices);

        for (int i = 0; i < numBatches - 1; i++)
        {
            var covariate = covariates.index_select(0, i);
            var spikeStart = sampleIndices[i].item<long>();
            var spikeEnd = sampleIndices[i + 1].item<long>();
            var spikes = spikingData.index_select(0, arange(spikeStart, spikeEnd));
            sortedSpikeEncoder.Encode(covariate, spikes);
            covariate = covariate.expand([spikes.size(0), -1]);
            sortedSpikeEncoder2.Encode(covariate, spikes);
        }

        var encoderIntensities = sortedSpikeEncoder.Intensities;
        var encoderIntensities2 = sortedSpikeEncoder2.Intensities;

        foreach (var (intensity1, intensity2) in encoderIntensities.Zip(encoderIntensities2))
        {
            Assert.IsTrue(intensity1.allclose(intensity2));
        }
    }

        [TestMethod]
    public void TestClusterlessMarksEncoderDensity2DCovariateDownsampled()
    {
        var estimationMethod = Core.Estimation.EstimationMethod.KernelDensity;
        double[] covariateBandwidth = [5.0, 5.0];
        var numDimensions = 2;
        long[] evaluationSteps = [50, 50];
        var steps = 200;
        var cycles = 10;
        double[] min = [0, 0];
        double[] max = [100, 100];
        var markDimensions = 4;
        var numChannels = 8;
        double[] markBandwidth = [1.0, 1.0, 1.0, 1.0];
        var numNeurons = 40;
        var placeFieldRadius = 8.0;
        var firingThreshold = 0.2;
        var spikeScale = 5.0;
        var noiseScale = 0.5;
        var scalarType = ScalarType.Float32;
        Device device = CPU;

        var stateSpace = new Core.StateSpace.DiscreteUniform(
            numDimensions,
            min,
            max,
            evaluationSteps,
            device: device,
            scalarType: scalarType
        );

        var clusterlessMarksEncoder = new Core.Encoder.ClusterlessMarks(
            estimationMethod, 
            covariateBandwidth,
            markDimensions,
            numChannels,
            markBandwidth,
            stateSpace,
            device: device,
            scalarType: scalarType
        );

        var (position2D, spikingData) = Simulation.Utilities.InitializeSimulation2D(
            steps,
            cycles,
            min[0],
            max[0],
            min[1],
            max[1],
            numNeurons,
            placeFieldRadius,
            firingThreshold
        );

        var marksData = Simulation.Simulate.MarksAtPosition(
            position2D,
            spikingData,
            markDimensions,
            numChannels,
            spikeScale: spikeScale,
            noiseScale: noiseScale,
            scalarType: scalarType,
            device: device
        );

        var numSamples = position2D.size(0);
        var sampleIndices = arange(0, numSamples, 10, dtype: int64);
        var numBatches = sampleIndices.size(0);
        var covariates = position2D.index_select(0, sampleIndices);

        for (int i = 0; i < numBatches - 1; i++)
        {
            var covariate = covariates.index_select(0, i);
            var marksStart = sampleIndices[i].item<long>();
            var marksEnd = sampleIndices[i + 1].item<long>();
            var marks = marksData.index_select(0, arange(marksStart, marksEnd));
            clusterlessMarksEncoder.Encode(covariate, marks);
        }

        var encoderIntensities = clusterlessMarksEncoder.Intensities;

        var clusterlessMarksEncoder2 = new Core.Encoder.ClusterlessMarks(
            estimationMethod, 
            covariateBandwidth,
            markDimensions,
            numChannels,
            markBandwidth,
            stateSpace,
            device: device,
            scalarType: scalarType
        );

        for (int i = 0; i < numBatches - 1; i++)
        {
            var covariate = covariates.index_select(0, i);
            var marksStart = sampleIndices[i].item<long>();
            var marksEnd = sampleIndices[i + 1].item<long>();
            var marks = marksData.index_select(0, arange(marksStart, marksEnd));
            covariate = covariate.expand([marks.size(0), -1]);
            clusterlessMarksEncoder2.Encode(covariate, marks);
        }

        var encoderIntensities2 = clusterlessMarksEncoder2.Intensities;

        foreach (var (intensity1, intensity2) in encoderIntensities.Zip(encoderIntensities2))
        {
            Assert.IsTrue(intensity1.allclose(intensity2));
        }
    }
}