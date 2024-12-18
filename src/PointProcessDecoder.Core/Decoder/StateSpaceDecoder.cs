using static TorchSharp.torch;

using PointProcessDecoder.Core.Transitions;
using PointProcessDecoder.Core.Likelihood;

namespace PointProcessDecoder.Core.Decoder;

public class StateSpaceDecoder : PointProcessDecoder
{
    private readonly Device _device;
    public override Device Device => _device;

    private readonly ScalarType _scalarType;
    public override ScalarType ScalarType => _scalarType;

    private readonly Tensor _initialState;
    public Tensor InitialState => _initialState;

    private readonly IStateTransitions _stateTransitions;
    public IStateTransitions Transitions => _stateTransitions;

    private Tensor _posterior;
    public Tensor Posterior => _posterior;

    private readonly IStateSpace _stateSpace;
    private readonly Func<Tensor, Tensor, Tensor> _likelihoodMethod;

    public StateSpaceDecoder(
        TransitionsType transitionsType,
        LikelihoodType likelihoodType,
        IStateSpace stateSpace,
        double[]? sigmaRandomWalk = null,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
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

        _likelihoodMethod = likelihoodType switch
        {
            LikelihoodType.Poisson => PoissonLikelihood.LogLikelihood,
            _ => throw new ArgumentException("Invalid likelihood type.")
        };

        var n = _stateSpace.Points.shape[0];
        _initialState = ones(n, dtype: _scalarType, device: _device) / n;
        _posterior = _initialState.clone();
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
    public override Tensor Decode(Tensor inputs, IEnumerable<Tensor> conditionalIntensities)
    {
        using var _ = NewDisposeScope();

        var conditionalIntensitiesTensor = stack(conditionalIntensities.Select(ci => ci.flatten()), dim: -1);
        var logLikelihood = _likelihoodMethod(inputs, conditionalIntensitiesTensor);
        var outputShape = new long[] { inputs.shape[0] }
            .Concat(_stateSpace.Shape)
            .ToArray();
        var output = zeros(outputShape, dtype: _scalarType, device: _device);
        for (int i = 0; i < inputs.shape[0]; i++)
        {
            var posteriorUpdated = _stateTransitions.Transitions.matmul(_posterior);
            posteriorUpdated /= posteriorUpdated.sum();
            logLikelihood[i] -= logLikelihood[i].max() + posteriorUpdated.log().max();
            _posterior = exp(logLikelihood[i] * inputs[i].any() + posteriorUpdated.log());
            _posterior /= _posterior.sum();
            _posterior = _posterior.nan_to_num().clamp_min(1e-12);
            output[i] = _posterior.reshape(_stateSpace.Shape);
        }
        _posterior.MoveToOuterDisposeScope();
        return output.MoveToOuterDisposeScope();
    }
}
