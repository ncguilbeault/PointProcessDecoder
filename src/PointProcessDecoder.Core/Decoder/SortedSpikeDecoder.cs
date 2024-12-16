using static TorchSharp.torch;

using PointProcessDecoder.Core.Transitions;

namespace PointProcessDecoder.Core.Decoder;

public class SortedSpikeDecoder : IDecoder
{
    private readonly Device _device;
    public Device Device => _device;

    private readonly ScalarType _scalarType;
    public ScalarType ScalarType => _scalarType;

    private readonly Tensor _initialState;
    public Tensor InitialState => _initialState;

    private readonly IStateTransitions _stateTransitions;
    public IStateTransitions Transitions => _stateTransitions;

    private Tensor _posterior;
    public Tensor Posterior => _posterior;

    private int _latentDimensions;

    public SortedSpikeDecoder(
        TransitionsType transitionsType,
        int latentDimensions,
        double[] minObservationSpace,
        double[] maxObservationSpace,
        long[] stepsObservationSpace,
        double[]? sigmaRandomWalk = null,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _latentDimensions = latentDimensions;

        _stateTransitions = transitionsType switch
        {
            TransitionsType.Uniform => new UniformTransitions(
                _latentDimensions, 
                minObservationSpace, 
                maxObservationSpace, 
                stepsObservationSpace, 
                device: _device,
                scalarType: _scalarType
            ),
            TransitionsType.RandomWalk => new RandomWalkTransitions(
                _latentDimensions, 
                minObservationSpace, 
                maxObservationSpace, 
                stepsObservationSpace, 
                sigmaRandomWalk, 
                device: _device,
                scalarType: _scalarType
            ),
            _ => throw new ArgumentException("Invalid transitions type.")
        };

        var points = _stateTransitions.Points;
        var n = points.shape[0];
        _initialState = ones(n, dtype: _scalarType, device: _device) / n;
        _posterior = _initialState.clone();
    }

    /// <summary>
    /// Decodes the input into the latent space using a bayesian point process decoder.
    /// Input tensor should be of shape (m, n) where m is the number of observations and n is the number of units.
    /// Conditional intensities should be a list of tensors which can be used to compute the likelihood of the observations given the inputs.
    /// </summary>
    /// <param name="inputs"></param>
    /// <param name="conditionalIntensities"></param>
    /// <returns>
    /// A tensor of shape (m, p) where m is the number of observations and p is the number of points in the latent space.
    /// </returns>
    public Tensor Decode(Tensor inputs, IEnumerable<Tensor> conditionalIntensities)
    {
        using var _ = NewDisposeScope();
        inputs = inputs.to_type(_scalarType).to(_device);

        var conditionalIntensitiesTensor = stack(conditionalIntensities.Select(ci => ci.flatten()), dim: -1)
            .to_type(_scalarType)
            .to(_device);

        var logLikelihood = LogLikelihood(inputs, conditionalIntensitiesTensor);
        var output = zeros_like(logLikelihood);
        for (int i = 0; i < inputs.shape[0]; i++)
        {
            _posterior = exp(logLikelihood[i] * inputs[i].any() + _stateTransitions.Transitions.T.matmul(_posterior).log().clamp(1e-10));
            _posterior /= _posterior.sum();
            output[i] = _posterior;
        }
        _posterior.MoveToOuterDisposeScope();
        return output.MoveToOuterDisposeScope();
    }

    private static Tensor LogLikelihood(Tensor inputs, Tensor conditionalIntensities)
    {
        using var _ = NewDisposeScope();
        conditionalIntensities = conditionalIntensities.clamp(1e-10);
        var logLikelihood = xlogy(inputs.unsqueeze(1), conditionalIntensities.unsqueeze(0)) - conditionalIntensities.unsqueeze(0);
        return logLikelihood.sum(dim: -1).MoveToOuterDisposeScope();
    }
}
