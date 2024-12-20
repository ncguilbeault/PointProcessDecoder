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

    // Define a function to compute a random walk transition matrix (gaussian) across the latent space. 
    // It should compute this efficiently with tensor broadcasting and it should be able to handle multiple dimensions.
    private static Tensor ComputeRandomWalkTransitions(
        IStateSpace stateSpace,
        Device device,
        ScalarType scalarType,
        Tensor? sigma = null)
    {
        using var _ = NewDisposeScope();
        var points = stateSpace.Points;

        var dist = cdist(points, points);
        var bandwidth = sigma is null ? dist.mean() / 2 : sigma;
        var weights = exp(-dist.pow(2) / (2 * bandwidth.pow(2)));
        var transitions = weights / weights.sum(1, true);

        return transitions
            .MoveToOuterDisposeScope();
    }
}
