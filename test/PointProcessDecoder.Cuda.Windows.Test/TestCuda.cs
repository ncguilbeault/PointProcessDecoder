using TorchSharp;
using static TorchSharp.torch;
using PointProcessDecoder.Core;
using PointProcessDecoder.Simulation;

namespace PointProcessDecoder.Cuda.Windows.Test;

[TestClass]
public class TestCuda
{
    [TestMethod]
    public void InitializeCudaDevice()
    {
        var device = CUDA;
        device = InitializeDevice(device);
        Assert.IsTrue(device.type == DeviceType.CUDA);
    }

    [TestMethod]
    public void TestCudaDevice()
    {
        var cudaAvailable = cuda.is_available();
        var deviceCount = cuda.device_count();
        var cudnnAvailable = cuda.is_cudnn_available();
        Assert.IsTrue(cudaAvailable && deviceCount > 0 && cudnnAvailable);
    }

    [TestMethod]
    public void TestLoadModelOnCuda()
    {
        var device = CUDA;
        InitializeDevice(device);
        Assert.IsTrue(device.type == DeviceType.CUDA);

        var position = Simulate.SinPosition(
            200, 
            10, 
            0, 
            100,
            device: device
        );

        var placeFieldCenters = Simulate.PlaceFieldCenters(
            0, 
            100, 
            40,
            device: device
        );

        var spikingData = Simulate.SpikesAtPosition(
            position, 
            placeFieldCenters,
            8.0, 
            0.2, 
            seed: 0,
            device: device
        );

        var model = new PointProcessModel(
            estimationMethod: Core.Estimation.EstimationMethod.KernelDensity,
            transitionsType: Core.Transitions.TransitionsType.RandomWalk,
            encoderType: Core.Encoder.EncoderType.SortedSpikes,
            decoderType: Core.Decoder.DecoderType.StateSpaceDecoder,
            stateSpaceType: Core.StateSpace.StateSpaceType.DiscreteUniform,
            likelihoodType: Core.Likelihood.LikelihoodType.Poisson,
            minStateSpace: [0],
            maxStateSpace: [100],
            stepsStateSpace: [50],
            covariateBandwidth: [1],
            stateSpaceDimensions: 1,
            numUnits: 40,
            sigmaRandomWalk: 1,
            device: device
        );

        model.Encode(position, spikingData);

        var prediction = model.Decode(spikingData);

        model.Save("TestLoadModelOnCuda");

        var loadedModel = PointProcessModel.Load("TestLoadModelOnCuda", device) as PointProcessModel;

        Assert.IsNotNull(loadedModel);

        Assert.IsTrue(loadedModel.Device.type == DeviceType.CUDA);
        Assert.IsTrue(loadedModel.Encoder.Estimations[0].Device.type == DeviceType.CUDA);
        Assert.IsTrue(loadedModel.Encoder.Estimations[0].Kernels.device.type == DeviceType.CUDA);

        var loadedPrediction = loadedModel.Decode(spikingData);

        var sameValues = prediction
            .eq(loadedPrediction)
            .all()
            .item<bool>();
            
        Assert.IsTrue(sameValues);
    }
}