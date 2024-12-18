using TorchSharp;
using static TorchSharp.torch;
using PointProcessDecoder.Core;

namespace PointProcessDecoder.Cuda.Linux.Test;

[TestClass]
public class TestCuda
{
    [TestMethod]
    public void InitializeCudaDevice()
    {
        var device = CUDA;
        InitializeDevice(device);
        Assert.IsTrue(device.type == DeviceType.CUDA);
    }

    [TestMethod]
    public void TestCudaDevice()
    {
        var cudaAvailable = cuda.is_available();
        var deviceCount = cuda.device_count();
        var cudnnAvailable = cuda.is_cudnn_available();
    }
}