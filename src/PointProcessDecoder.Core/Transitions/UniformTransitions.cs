using TorchSharp.Modules;
using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Transitions;

public class UniformTransitions : IStateTransitions
{
    private readonly Device _device;
    /// <inheritdoc/>
    public Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <inheritdoc/>
    public ScalarType ScalarType => _scalarType;

    public TransitionsType TransitionsType => TransitionsType.Uniform;

    private readonly Tensor _transitions;
    /// <inheritdoc/>
    public Tensor Transitions => _transitions;

    private readonly IStateSpace _stateSpace;

    public UniformTransitions(
        IStateSpace stateSpace,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _stateSpace = stateSpace;

        _transitions = ComputeUniformTransitions(
            _stateSpace,
            _device,
            _scalarType
        );
    }

    private static Tensor ComputeUniformTransitions(
        IStateSpace stateSpace,
        Device device,
        ScalarType scalarType
    )
    {
        using var _ = NewDisposeScope();
        var n = stateSpace.Points.shape[0];
        var transitions = ones(n, n, device: device, dtype: scalarType);
        transitions /= transitions.sum(1, true);
        return transitions
            .to_type(scalarType)
            .to(device)
            .MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _transitions.Dispose();
    }
}
