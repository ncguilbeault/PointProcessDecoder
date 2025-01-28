using TorchSharp.Modules;
using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Transitions;

public class RandomWalkTransitions : IStateTransitions
{
    private readonly Device _device;
    /// <inheritdoc/>
    public Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <inheritdoc/>
    public ScalarType ScalarType => _scalarType;

    /// <inheritdoc/>
    public TransitionsType TransitionsType => TransitionsType.RandomWalk;

    private readonly Tensor _transitions;
    /// <inheritdoc/>
    public Tensor Transitions => _transitions;

    private readonly Tensor? _sigma;
    /// <summary>
    /// The standard deviation of the Gaussian transitions.
    /// </summary>
    public Tensor? Sigma => _sigma;

    private readonly IStateSpace _stateSpace;

    public RandomWalkTransitions(
        IStateSpace stateSpace,
        double? sigma = null, 
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _stateSpace = stateSpace;
        _sigma = sigma is not null ? tensor(sigma.Value, device: _device, dtype: _scalarType) : null;

        _transitions = ComputeRandomWalkTransitions(
            _stateSpace,
            _device,
            _scalarType, 
            _sigma
        );
    }

    private static Tensor ComputeRandomWalkTransitions(
        IStateSpace stateSpace,
        Device device,
        ScalarType scalarType,
        Tensor? sigma = null)
    {
        using var _ = NewDisposeScope();
        var points = stateSpace.Points;

        var dist = points.unsqueeze(0) - points.unsqueeze(1);
        var bandwidth = sigma is null ? dist.mean() / 2 : sigma;
        var sumSquaredDiff = dist
            .pow(exponent: 2)
            .sum(dim: 2);
        var estimate = exp(-0.5 * sumSquaredDiff / bandwidth);
        var weights = estimate / sqrt(pow(2 * Math.PI, stateSpace.Dimensions) * bandwidth);
        var transitions = weights / points.shape[1];

        return transitions
            .to_type(type: scalarType)
            .to(device: device)
            .MoveToOuterDisposeScope();
    }

    public void Dispose()
    {
        _transitions.Dispose();
        _sigma?.Dispose();
    }
}
