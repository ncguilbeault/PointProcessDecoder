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
        double[] minLatentSpace,
        double[] maxLatentSpace,
        long[] stepsLatentSpace,
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
                minLatentSpace, 
                maxLatentSpace, 
                stepsLatentSpace, 
                device: _device,
                scalarType: _scalarType
            ),
            TransitionsType.RandomWalk => new RandomWalkTransitions(
                _latentDimensions, 
                minLatentSpace, 
                maxLatentSpace, 
                stepsLatentSpace, 
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
        inputs = inputs.to_type(_scalarType).to(_device);

        using var conditionalIntensitiesTensor = stack(conditionalIntensities.Select(t => t.flatten()), dim: -1)
            .to_type(_scalarType)
            .to(_device);

        using var logLikelihood = LogLikelihood(inputs, conditionalIntensitiesTensor);
        _posterior = exp(logLikelihood + _stateTransitions.Transitions.matmul(_posterior).log());
        
        return _posterior / _posterior.sum(0, true);
    }

    private static Tensor LogLikelihood(Tensor inputs, Tensor conditionalIntensities)
    {
        using var _ = NewDisposeScope();
        var clippedConditionalIntensities = conditionalIntensities.clamp(1e-10);
        var logLikelihood = xlogy(inputs.unsqueeze(1), clippedConditionalIntensities.unsqueeze(0)).sum(dim: 2);
        return (logLikelihood - logLikelihood.logsumexp(1, true)).MoveToOuterDisposeScope();
    }
}
