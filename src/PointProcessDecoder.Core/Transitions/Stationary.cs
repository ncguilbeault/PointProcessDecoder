using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Transitions;

/// <summary>
/// Represents stationary state transitions.
/// </summary>
public class Stationary : ModelComponent, IStateTransitions
{
    private readonly Device _device;
    /// <inheritdoc/>
    public override Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <inheritdoc/>
    public override ScalarType ScalarType => _scalarType;

    /// <inheritdoc/>
    public TransitionsType TransitionsType => TransitionsType.Stationary;

    private readonly Tensor _transitions;
    /// <inheritdoc/>
    public Tensor Transitions => _transitions;

    private readonly IStateSpace _stateSpace;

    public Stationary(
        IStateSpace stateSpace,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _stateSpace = stateSpace;

        _transitions = ComputeStationary(
            _stateSpace,
            _device,
            _scalarType
        );
    }

    private static Tensor ComputeStationary(
        IStateSpace stateSpace,
        Device device,
        ScalarType scalarType
    )
    {
        using var _ = NewDisposeScope();
        var points = stateSpace.Points;
        var n = stateSpace.Points.size(0);
        var transitions = eye(n, device: device, dtype: scalarType);
        return transitions
            .to_type(type: scalarType)
            .to(device: device)
            .MoveToOuterDisposeScope();
    }
}
