using TorchSharp.Modules;
using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Transitions;

public class UniformTransitions : StateTransitions
{
    private readonly int _dimensions;
    /// <inheritdoc/>
    public override int Dimensions => _dimensions;

    private readonly Tensor _points;
    /// <inheritdoc/>    
    public override Tensor Points => _points;

    private readonly Device _device;
    /// <inheritdoc/>
    public override Device Device => _device;

    private readonly Tensor _values;
    /// <inheritdoc/>
    public override Tensor Values => _values;

    /// <summary>
    /// Create a new instance of the <see cref="UniformTransitions"/> class.
    /// </summary>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <param name="steps"></param>
    /// <param name="device"></param>
    public UniformTransitions(double min, double max, long steps, Device? device = null)
    {
        _dimensions = 1;
        _device = device ?? CPU;
        _points = ComputeLatentSpace(
            [min], 
            [max],
            [steps],
            _device
        );
        _values = ComputeUniformTransitions(_points);
    }

    /// <summary>
    /// Create a new instance of the <see cref="UniformTransitions"/> class.
    /// </summary>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <param name="steps"></param>
    /// <param name="device"></param>
    /// <exception cref="ArgumentException"></exception>
    public UniformTransitions(int dimensions, double[] min, double[] max, long[] steps, Device? device = null)
    {
        if (dimensions != min.Length || dimensions != max.Length || dimensions != steps.Length)
        {
            throw new ArgumentException("The lengths of min, max, and steps must be equal.");
        }

        _dimensions = dimensions;
        _device = device ?? CPU;
        _points = ComputeLatentSpace(
            min, 
            max,
            steps,
            _device
        );
        _values = ComputeUniformTransitions(_points);
    }

    private Tensor ComputeUniformTransitions(Tensor points)
    {
        using (var _ = NewDisposeScope())
        {
            var n = points.shape[0];
            var transitions = ones(n, n, device: _device);
            transitions /= transitions.sum(1, true);
            return transitions.MoveToOuterDisposeScope();
        }
    }
}
