using static TorchSharp.torch;

namespace PointProcessDecoder.Core.StateSpace;

/// <summary>
/// Represents a discrete uniform state space.
/// </summary>
public class DiscreteUniformStateSpace : ModelComponent, IStateSpace
{
    private readonly Device _device;
    /// <inheritdoc/>
    public override Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <inheritdoc/>
    public override ScalarType ScalarType => _scalarType;

    /// <inheritdoc/>
    public StateSpaceType StateSpaceType => StateSpaceType.DiscreteUniformStateSpace;

    private readonly int _dimensions;
    /// <inheritdoc/>
    public int Dimensions => _dimensions;

    private readonly Tensor _points;
    /// <inheritdoc/>
    public Tensor Points => _points;

    private readonly long[] _shape;
    /// <inheritdoc/>
    public long[] Shape => _shape;

    /// <summary>
    /// Initializes a new instance of the <see cref="DiscreteUniformStateSpace"/> class.
    /// </summary>
    /// <param name="dimensions"></param>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <param name="steps"></param>
    /// <param name="device"></param>
    /// <param name="scalarType"></param>
    /// <exception cref="ArgumentException"></exception>
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

    /// <inheritdoc/>
    public override void Dispose()
    {
        _points.Dispose();
    }
}
