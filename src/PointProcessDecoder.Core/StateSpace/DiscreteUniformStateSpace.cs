using static TorchSharp.torch;

namespace PointProcessDecoder.Core.StateSpace;

public class DiscreteUniformStateSpace : IStateSpace
{
    private readonly Device _device;
    /// <summary>
    /// The device on which the tensor is stored.
    /// </summary>
    public Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <summary>
    /// The scalar type of the tensor.
    /// </summary>
    public ScalarType ScalarType => _scalarType;

    private readonly int _dimensions;
    /// <summary>
    /// The number of dimensions in the latent space.
    /// </summary>
    public int Dimensions => _dimensions;

    private readonly Tensor _points;
    /// <summary>
    /// The points in the state space. Points should be in the form of a tensor with shape (n, d) where n is the number of points and d is the dimensionality of the latent space.
    /// </summary>
    public Tensor Points => _points;

    private readonly long[] _shape;
    /// <summary>
    /// The shape of the state space.
    /// </summary>
    public long[] Shape => _shape;

    public DiscreteUniformStateSpace(
        int dimensions,
        double[] min,
        double[] max,
        long[] steps,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        if (dimensions != min.Length || dimensions != max.Length || dimensions != steps.Length)
        {
            throw new ArgumentException("The lengths of min, max, and steps must be equal to the number of dimensions.");
        }

        _dimensions = dimensions;
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _points = ComputeDiscreteUniformStateSpace(min, max, steps, _device, _scalarType);
        _shape = steps;
    }

    /// <summary>
    /// Method to compute the latent space given the minimum, maximum, and steps.
    /// </summary>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <param name="steps"></param>
    /// <param name="device"></param>
    /// <returns></returns>
    public static Tensor ComputeDiscreteUniformStateSpace(
        double[] min, 
        double[] max, 
        long[] steps,
        Device device,
        ScalarType scalarType
    )
    {
        var dims = min.Length;
        using var _ = NewDisposeScope();
        var arrays = new Tensor[dims];
        for (int i = 0; i < dims; i++)
        {
            arrays[i] = linspace(min[i], max[i], steps[i]);
        }
        var grid = meshgrid(arrays);
        var gridSpace = vstack(grid.Select(tensor => tensor.flatten()).ToList()).T;
        return gridSpace
            .to_type(scalarType)
            .to(device)
            .MoveToOuterDisposeScope();
    }
}
