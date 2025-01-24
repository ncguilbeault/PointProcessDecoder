using static TorchSharp.torch;

using PointProcessDecoder.Core.Transitions;
using PointProcessDecoder.Core.Likelihood;

namespace PointProcessDecoder.Core.Decoder;

public class StateSpaceDecoder : IDecoder
{
    private readonly Device _device;
    public Device Device => _device;

    private readonly ScalarType _scalarType;
    public ScalarType ScalarType => _scalarType;

    private readonly Tensor _initialState = empty(0);
    public Tensor InitialState => _initialState;

    private readonly IStateTransitions _stateTransitions;
    public IStateTransitions Transitions => _stateTransitions;

    private Tensor _posterior;
    public Tensor Posterior => _posterior;

    private readonly IStateSpace _stateSpace;
    private readonly double _eps;

    public StateSpaceDecoder(
        TransitionsType transitionsType,
        IStateSpace stateSpace,
        double? sigmaRandomWalk = null,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _eps = finfo(_scalarType).eps;
        _stateSpace = stateSpace;

        _stateTransitions = transitionsType switch
        {
            TransitionsType.Uniform => new UniformTransitions(
                _stateSpace,
                device: _device,
                scalarType: _scalarType
            ),
            TransitionsType.RandomWalk => new RandomWalkTransitions(
                _stateSpace,
                sigmaRandomWalk, 
                device: _device,
                scalarType: _scalarType
            ),
            _ => throw new ArgumentException("Invalid transitions type.")
        };

        var n = _stateSpace.Points.shape[0];
        _initialState = ones(n, dtype: _scalarType, device: _device) / n;
        _posterior = empty(0);
    }

    /// <summary>
    /// Decodes the input into the latent space using a bayesian state space decoder.
    /// Input tensor should be of shape (m, n) where m is the number of observations and n is the number of units.
    /// Conditional intensities should be a list of tensors which can be used to compute the likelihood of the observations given the inputs.
    /// </summary>
    /// <param name="inputs"></param>
    /// <param name="conditionalIntensities"></param>
    /// <returns>
    /// A tensor of shape (m, p) where m is the number of observations and p is the number of points in the latent space.
    /// </returns>
    public Tensor Decode(Tensor inputs, Tensor likelihood)
    {
        using var _ = NewDisposeScope();

        var outputShape = new long[] { inputs.shape[0] }
            .Concat(_stateSpace.Shape)
            .ToArray();

        var output = zeros(outputShape, dtype: _scalarType, device: _device);

        if (_posterior.numel() == 0) {
            _posterior = _initialState * likelihood[0].flatten();
            _posterior /= _posterior.sum();
            output[0] = _posterior.reshape(_stateSpace.Shape);
        }

        for (int i = 1; i < inputs.shape[0]; i++)
        {
            var update = _stateTransitions.Transitions.matmul(_posterior)
                .nan_to_num()
                .log();
            _posterior = exp(likelihood[i].flatten() + update)
                .clamp_min(_eps);
            _posterior /= _posterior.sum();
            output[i] = _posterior.reshape(_stateSpace.Shape);
        }
        _posterior.MoveToOuterDisposeScope();
        return output.MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _stateTransitions.Dispose();
        _initialState.Dispose();
        _posterior.Dispose();
    }
}
