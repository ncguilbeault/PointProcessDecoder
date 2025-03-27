using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Transitions;

/// <summary>
/// Represents uniform state transitions.
/// </summary>
public class Uniform : ModelComponent, IStateTransitions
{
    private readonly Device _device;
    /// <inheritdoc/>
    public override Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <inheritdoc/>
    public override ScalarType ScalarType => _scalarType;

    /// <inheritdoc/>
    public TransitionsType TransitionsType => TransitionsType.Uniform;

    private readonly Tensor _transitions;
    /// <inheritdoc/>
    public Tensor Transitions => _transitions;

    private readonly IStateSpace _stateSpace;

    /// <summary>
    /// Initializes a new instance of the <see cref="Uniform"/> class.
    /// </summary>
    /// <param name="stateSpace"></param>
    /// <param name="device"></param>
    /// <param name="scalarType"></param>
    public Uniform(
        IStateSpace stateSpace,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _stateSpace = stateSpace;

        _transitions = ComputeUniform(
            _stateSpace,
            _device,
            _scalarType
        );
    }

    private static Tensor ComputeUniform(
        IStateSpace stateSpace,
        Device device,
        ScalarType scalarType
    )
    {
        using var _ = NewDisposeScope();
        var n = stateSpace.Points.size(0);
        var transitions = ones(n, n, device: device, dtype: scalarType);
        transitions /= transitions.sum(1, true);
        return transitions
            .to_type(scalarType)
            .to(device)
            .MoveToOuterDisposeScope();
    }
}
