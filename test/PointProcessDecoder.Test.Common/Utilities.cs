using static TorchSharp.torch;
using PointProcessDecoder.Simulation;
using PointProcessDecoder.Plot;
using PointProcessDecoder.Core;

namespace PointProcessDecoder.Test.Common;

public static class Utilities
{
    public static Tensor ReadBinaryFile(
        string binary_file,
        Device? device = null,
        ScalarType scalarType = ScalarType.Float32
    )
    {
        device ??= CPU;
        byte[] fileBytes = File.ReadAllBytes(binary_file);
        int elementCount = fileBytes.Length / sizeof(double);
        double[] doubleArray = new double[elementCount];
        Buffer.BlockCopy(fileBytes, 0, doubleArray, 0, fileBytes.Length);
        Tensor t = tensor(doubleArray, device: device, dtype: scalarType);
        return t;
    }

    public static (Tensor, Tensor) InitializeRealClusterlessMarksData(
        string positionFile,
        string marksFile,
        Device? device = null,
        ScalarType scalarType = ScalarType.Float32
    )
    {
        var position = ReadBinaryFile(positionFile, device, scalarType);
        var marks = ReadBinaryFile(marksFile, device, scalarType);
        return (position, marks);
    }

    public static (Tensor, Tensor) InitializeRealSortedSpikeData(
        string positionFile,
        string spikesFile,
        Device? device = null,
        ScalarType scalarType = ScalarType.Float32
    )
    {
        var position = ReadBinaryFile(positionFile, device, scalarType);
        var spikes = ReadBinaryFile(spikesFile, device, scalarType);
        return (position, spikes);
    }
}
