using TorchSharp.Modules;
using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Transitions;

public class RandomWalkTransitions : StateTransitions
{
    private readonly Tensor _points;
    /// <inheritdoc/>    
    public override Tensor Points => _points;

    private readonly int _dimensions;
    /// <inheritdoc/>
    public override int Dimensions => _dimensions;

    private readonly Device _device;
    /// <inheritdoc/>
    public override Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <inheritdoc/>
    public override ScalarType ScalarType => _scalarType;

    private readonly Tensor _transitions;
    /// <inheritdoc/>
    public override Tensor Transitions => _transitions;

    private readonly Tensor? _sigma;
    /// <summary>
    /// The standard deviation of the Gaussian transitions.
    /// </summary>
    public Tensor? Sigma => _sigma;

    public RandomWalkTransitions(
        double min, 
        double max, 
        long steps, 
        double? sigma = null,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        _dimensions = 1;
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _sigma = sigma is not null ? tensor(new double[] { sigma.Value }, device: _device, dtype: _scalarType) : null;
        
        _points = ComputeLatentSpace(
            [min], 
            [max],
            [steps],
            _device,
            _scalarType
        );

        _transitions = ComputeRandomWalkTransitions(
            _points,
            _device,
            _scalarType, 
            _sigma
        );
    }

    public RandomWalkTransitions(
        int dimensions, 
        double[] min, 
        double[] max, 
        long[] steps, 
        double[]? sigma = null, 
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        if (dimensions != min.Length || dimensions != max.Length || dimensions != steps.Length)
        {
            throw new ArgumentException("The lengths of min, max, and steps must be equal to the number of dimensions.");
        }

        if (sigma is not null && sigma.Length != dimensions)
        {
            throw new ArgumentException("The length of sigma must be equal to the number of dimensions.");
        }

        _dimensions = dimensions;
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _sigma = sigma is not null ? tensor(sigma, device: _device, dtype: _scalarType) : null;

        _points = ComputeLatentSpace(
            min, 
            max,
            steps,
            _device,
            _scalarType
        );

        _transitions = ComputeRandomWalkTransitions(
            _points,
            _device,
            _scalarType, 
            _sigma
        );
    }

    // Define a function to compute a random walk transition matrix (gaussian) across the latent space. 
    // It should compute this efficiently with tensor broadcasting and it should be able to handle multiple dimensions.
    private static Tensor ComputeRandomWalkTransitions(
        Tensor points,
        Device device,
        ScalarType scalarType,
        Tensor? sigma = null)
    {
        using var _ = NewDisposeScope();

        if (sigma is not null)
            points /= sigma;

        var dist = cdist(points, points);
        var bandwidth = sigma is not null ? dist.mean() / 2 : tensor(1, device: device, dtype: scalarType);
        var weights = exp(-dist.pow(2) / (2 * bandwidth.pow(2)));
        var transitions = weights / weights.sum(1, true);

        return transitions
            .MoveToOuterDisposeScope();
    }
}
