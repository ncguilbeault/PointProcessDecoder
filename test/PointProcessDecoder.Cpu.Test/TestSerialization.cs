using static TorchSharp.torch;

using PointProcessDecoder.Core;
using PointProcessDecoder.Simulation;

namespace PointProcessDecoder.Cpu.Test;

[TestClass]
public class TestSerialization
{
    [TestMethod]
    public void TestModelSerialization()
    {
        var model = new PointProcessModel(
            estimationMethod: Core.Estimation.EstimationMethod.KernelDensity,
            transitionsType: Core.Transitions.TransitionsType.Uniform,
            encoderType: Core.Encoder.EncoderType.SortedSpikeEncoder,
            decoderType: Core.Decoder.DecoderType.StateSpaceDecoder,
            stateSpaceType: Core.StateSpace.StateSpaceType.DiscreteUniformStateSpace,
            likelihoodType: Core.Likelihood.LikelihoodType.Poisson,
            minStateSpace: [0],
            maxStateSpace: [100],
            stepsStateSpace: [50],
            observationBandwidth: [1],
            stateSpaceDimensions: 1,
            nUnits: 40,
            device: CPU,
            scalarType: ScalarType.Float64
        );

        var basePath = "TestModelSerialization";

        model.Save(basePath);

        // Check configuration exists
        Assert.IsTrue(Directory.Exists(basePath));
        Assert.IsTrue(Directory.GetFiles(basePath).Length > 0);
        Assert.IsTrue(File.Exists(Path.Combine(basePath, "configuration.json")));
        Assert.IsTrue(new FileInfo(Path.Combine(basePath, "configuration.json")).Length > 0);
    }

    [TestMethod]
    public void TestModelDeserialization()
    {
        var model = new PointProcessModel(
            estimationMethod: Core.Estimation.EstimationMethod.KernelDensity,
            transitionsType: Core.Transitions.TransitionsType.Uniform,
            encoderType: Core.Encoder.EncoderType.SortedSpikeEncoder,
            decoderType: Core.Decoder.DecoderType.StateSpaceDecoder,
            stateSpaceType: Core.StateSpace.StateSpaceType.DiscreteUniformStateSpace,
            likelihoodType: Core.Likelihood.LikelihoodType.Poisson,
            minStateSpace: [0, 0],
            maxStateSpace: [100, 100],
            stepsStateSpace: [50, 50],
            observationBandwidth: [1, 1],
            stateSpaceDimensions: 2,
            nUnits: 40,
            device: CPU,
            scalarType: ScalarType.Float64
        );

        var basePath = "TestModelDeserialization";
        model.Save(basePath);

        var loadedModel = PointProcessModel.Load(basePath) as PointProcessModel;

        Assert.IsNotNull(loadedModel);

        var sameValues = loadedModel.StateSpace.Points
            .eq(model.StateSpace.Points)
            .all()
            .item<bool>();

        Assert.IsTrue(sameValues);

        var sameShape = loadedModel.StateSpace.Points.shape
            .SequenceEqual(model.StateSpace.Points.shape);

        Assert.IsTrue(sameShape);
    }

    [TestMethod]
    public void TestSortedUnitsUniformDensityDeserialization()
    {
        var position = Simulate.Position(
            200, 
            10, 
            0, 
            100
        );

        var placeFieldCenters = Simulate.PlaceFieldCenters(
            0, 
            100, 
            40,
            seed: 0
        );

        var spikingData = Simulate.SpikesAtPosition(
            position, 
            placeFieldCenters,
            8.0, 
            0.2, 
            seed: 0
        );

        var model = new PointProcessModel(
            estimationMethod: Core.Estimation.EstimationMethod.KernelDensity,
            transitionsType: Core.Transitions.TransitionsType.Uniform,
            encoderType: Core.Encoder.EncoderType.SortedSpikeEncoder,
            decoderType: Core.Decoder.DecoderType.StateSpaceDecoder,
            stateSpaceType: Core.StateSpace.StateSpaceType.DiscreteUniformStateSpace,
            likelihoodType: Core.Likelihood.LikelihoodType.Poisson,
            minStateSpace: [0],
            maxStateSpace: [100],
            stepsStateSpace: [50],
            observationBandwidth: [1],
            stateSpaceDimensions: 1,
            nUnits: 40
        );

        model.Encode(position, spikingData);
        var prediction = model.Decode(spikingData);

        var basePath = "TestSortedUnitsUniformDensityDeserialization";
        model.Save(basePath);

        var loadedModel = PointProcessModel.Load(basePath) as PointProcessModel;

        Assert.IsNotNull(loadedModel);

        var predictionLoaded = loadedModel.Decode(spikingData);

        var sameValues = prediction
            .eq(predictionLoaded)
            .all()
            .item<bool>();

        Assert.IsTrue(sameValues);
    }

    [TestMethod]
    public void TestSortedUnitsRandomWalkCompressionDeserialization()
    {
        var position = Simulate.Position(
            200, 
            10, 
            0, 
            100
        );

        var placeFieldCenters = Simulate.PlaceFieldCenters(
            0, 
            100, 
            40,
            seed: 0
        );

        var spikingData = Simulate.SpikesAtPosition(
            position, 
            placeFieldCenters,
            8.0, 
            0.2, 
            seed: 0
        );

        var model = new PointProcessModel(
            estimationMethod: Core.Estimation.EstimationMethod.KernelCompression,
            transitionsType: Core.Transitions.TransitionsType.RandomWalk,
            encoderType: Core.Encoder.EncoderType.SortedSpikeEncoder,
            decoderType: Core.Decoder.DecoderType.StateSpaceDecoder,
            stateSpaceType: Core.StateSpace.StateSpaceType.DiscreteUniformStateSpace,
            likelihoodType: Core.Likelihood.LikelihoodType.Poisson,
            minStateSpace: [0],
            maxStateSpace: [100],
            stepsStateSpace: [50],
            observationBandwidth: [1],
            stateSpaceDimensions: 1,
            nUnits: 40,
            distanceThreshold: 1.5,
            sigmaRandomWalk: 1
        );

        model.Encode(position, spikingData);
        var prediction = model.Decode(spikingData);

        var basePath = "TestSortedUnitsRandomWalkCompressionDeserialization";
        model.Save(basePath);

        var loadedModel = PointProcessModel.Load(basePath) as PointProcessModel;

        Assert.IsNotNull(loadedModel);

        var predictionLoaded = loadedModel.Decode(spikingData);

        var sameValues = prediction
            .eq(predictionLoaded)
            .all()
            .item<bool>();

        Assert.IsTrue(sameValues);
    }

    [TestMethod]
    public void TestClusterlessRandomWalkDensityDeserializationAfterEncoding()
    {
        var position = Simulate.Position(
            200, 
            10, 
            0, 
            100
        );

        var placeFieldCenters = Simulate.PlaceFieldCenters(
            0, 
            100, 
            40,
            seed: 0
        );

        var spikingData = Simulate.SpikesAtPosition(
            position, 
            placeFieldCenters,
            8.0, 
            0.2, 
            seed: 0
        );

        var marks = Simulate.MarksAtPosition(
            position,
            spikingData, 
            4, 
            8, 
            seed: 0
        );

        var model = new PointProcessModel(
            estimationMethod: Core.Estimation.EstimationMethod.KernelDensity,
            transitionsType: Core.Transitions.TransitionsType.RandomWalk,
            encoderType: Core.Encoder.EncoderType.ClusterlessMarkEncoder,
            decoderType: Core.Decoder.DecoderType.StateSpaceDecoder,
            stateSpaceType: Core.StateSpace.StateSpaceType.DiscreteUniformStateSpace,
            likelihoodType: Core.Likelihood.LikelihoodType.Clusterless,
            minStateSpace: [0],
            maxStateSpace: [100],
            stepsStateSpace: [50],
            observationBandwidth: [1],
            stateSpaceDimensions: 1,
            markDimensions: 4,
            markChannels: 8,
            markBandwidth: [1, 1, 1, 1],
            sigmaRandomWalk: 1
        );

        model.Encode(position, marks);
        var prediction = model.Decode(marks);

        var basePath = "TestClusterlessRandomWalkDensityDeserializationAfterEncoding";
        model.Save(basePath);

        var loadedModel = PointProcessModel.Load(basePath) as PointProcessModel;

        Assert.IsNotNull(loadedModel);

        var predictionLoaded = loadedModel.Decode(marks);

        var sameValues = prediction
            .eq(predictionLoaded)
            .all()
            .item<bool>();

        Assert.IsTrue(sameValues);
    }

    [TestMethod]
    public void TestClusterlessUniformCompressionDeserializationAfterEncoding()
    {
        var position = Simulate.Position(
            200, 
            10, 
            0, 
            100
        );

        var placeFieldCenters = Simulate.PlaceFieldCenters(
            0, 
            100, 
            40,
            seed: 0
        );

        var spikingData = Simulate.SpikesAtPosition(
            position, 
            placeFieldCenters,
            8.0, 
            0.2, 
            seed: 0
        );

        var marks = Simulate.MarksAtPosition(
            position,
            spikingData, 
            4, 
            8, 
            seed: 0
        );

        var model = new PointProcessModel(
            estimationMethod: Core.Estimation.EstimationMethod.KernelCompression,
            transitionsType: Core.Transitions.TransitionsType.Uniform,
            encoderType: Core.Encoder.EncoderType.ClusterlessMarkEncoder,
            decoderType: Core.Decoder.DecoderType.StateSpaceDecoder,
            stateSpaceType: Core.StateSpace.StateSpaceType.DiscreteUniformStateSpace,
            likelihoodType: Core.Likelihood.LikelihoodType.Clusterless,
            minStateSpace: [0],
            maxStateSpace: [100],
            stepsStateSpace: [50],
            observationBandwidth: [1],
            stateSpaceDimensions: 1,
            markDimensions: 4,
            markChannels: 8,
            markBandwidth: [1, 1, 1, 1],
            distanceThreshold: 1.5
        );

        model.Encode(position, marks);
        var prediction = model.Decode(marks);

        var basePath = "TestClusterlessUniformCompressionDeserializationAfterEncoding";
        model.Save(basePath);

        var loadedModel = PointProcessModel.Load(basePath) as PointProcessModel;

        Assert.IsNotNull(loadedModel);

        var predictionLoaded = loadedModel.Decode(marks);

        var sameValues = prediction
            .eq(predictionLoaded)
            .all()
            .item<bool>();

        Assert.IsTrue(sameValues);
    }
}