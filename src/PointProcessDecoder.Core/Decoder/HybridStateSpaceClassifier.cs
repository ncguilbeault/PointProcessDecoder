using static TorchSharp.torch;

using PointProcessDecoder.Core.Transitions;

namespace PointProcessDecoder.Core.Decoder;

public class HybridStateSpaceClassifier : ModelComponent, IDecoder
{
    private readonly Device _device;
    /// <inheritdoc/>
    public override Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <inheritdoc/>
    public override ScalarType ScalarType => _scalarType;

    public DecoderType DecoderType => DecoderType.HybridStateSpaceClassifier;

    private const int NUM_STATES = 3;

    /// <summary>
    /// A 2x2 transition matrix for discrete states.
    /// </summary>
    private readonly Tensor _discreteTransitions = empty(0);

    /// <summary>
    /// The continuous transitions for each state, shape [2, n, n].
    /// </summary>
    private readonly Tensor _continuousTransitions = empty(0);

    /// <inheritdoc/>
    public Tensor[] Transitions => [_discreteTransitions, _continuousTransitions];

    /// <summary>
    /// The user-defined continuous state space.
    /// </summary>
    private readonly IStateSpace _stateSpace;

    private Tensor _posterior;
    /// <summary>
    /// The posterior distribution over the latent space.
    /// </summary>
    public Tensor Posterior => _posterior;

    private readonly Tensor _initialState = empty(0);
    /// <inheritdoc/>
    public Tensor InitialState => _initialState;

    private readonly double _eps;
    private readonly long[] _posteriorShape;
    private readonly double _stayProbability;


    public HybridStateSpaceClassifier(
        IStateSpace stateSpace,
        double? sigmaRandomWalk = null,
        double? stayProbability = null,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _eps = finfo(_scalarType).eps;
        _stateSpace = stateSpace;
        _stayProbability = stayProbability ?? 0.99;

        if (_stayProbability <= 0 || _stayProbability >= 1)
        {
            throw new ArgumentException("The stay probability must be between 0 and 1.");
        }

        var transitionProbability = (1 - _stayProbability) / (NUM_STATES - 1);

        _discreteTransitions = eye(
            NUM_STATES,
            dtype: _scalarType, 
            device: _device
        ) * (_stayProbability - transitionProbability);

        _discreteTransitions += ones(
            NUM_STATES, 
            NUM_STATES, 
            dtype: _scalarType, 
            device: _device
        ) * transitionProbability;

        _discreteTransitions = _discreteTransitions.reshape(NUM_STATES, NUM_STATES, 1, 1);

        var randomWalkTransitions = new RandomWalk(
            stateSpace: _stateSpace, 
            sigma: sigmaRandomWalk,
            device: _device,
            scalarType: _scalarType
        );

        var uniformTransitions = new Uniform(
            stateSpace: _stateSpace,
            device: _device,
            scalarType: _scalarType
        );

        var stationaryTransitions = new Stationary(
            stateSpace: _stateSpace,
            device: _device,
            scalarType: _scalarType
        );

        var n = _stateSpace.Points.size(0);
        _continuousTransitions = cat([
            stationaryTransitions.Transitions,
            randomWalkTransitions.Transitions,
            uniformTransitions.Transitions,
            stationaryTransitions.Transitions,
            randomWalkTransitions.Transitions,
            uniformTransitions.Transitions,
            uniformTransitions.Transitions,
            uniformTransitions.Transitions,
            uniformTransitions.Transitions
        ]).reshape(
            NUM_STATES, 
            NUM_STATES, 
            n, 
            n
        );
        
        _initialState = ones([NUM_STATES, n], dtype: _scalarType, device: _device) / (NUM_STATES * n);
        _posterior = empty(0);
        _posteriorShape = [.. new long[] { NUM_STATES }.Concat(_stateSpace.Shape)];
    }

    /// <inheritdoc/>
    public Tensor Decode(Tensor likelihood)
    {
        using var _ = NewDisposeScope();

        var outputShape = new long[] { likelihood.size(0) }
            .Concat(_posteriorShape);

        var output = zeros([.. outputShape], dtype: _scalarType, device: _device);

        var startIndex = 0;

        if (_posterior.NumberOfElements == 0) {
            _posterior = UpdatePosterior(_initialState, likelihood[0]);
            output[0] = _posterior.reshape(_posteriorShape);
            startIndex++;
        }

        long[] predShape = [.. new long[] { 1 }
            .Concat(_posterior.shape)
            .Concat([1])];

        for (int i = startIndex; i < likelihood.size(0); i++)
        {
            using var continuousPrediction = _continuousTransitions.matmul(_posterior.reshape(predShape));
            using var product = _discreteTransitions * continuousPrediction;
            using var newPosterior = product.sum([0, -1]);
            _posterior = UpdatePosterior(newPosterior, likelihood[i]);
            output[i] = _posterior.reshape(_posteriorShape);
        }
        
        _posterior.MoveToOuterDisposeScope();
        return output.MoveToOuterDisposeScope();
    }

    private Tensor UpdatePosterior(Tensor prior, Tensor likelihood)
    {
        var posterior = (prior * likelihood)
            .nan_to_num();

        if (posterior.sum().item<float>() == 0)
        {
            posterior = (_initialState * likelihood)
                .nan_to_num();
        }

        posterior /= posterior.sum();

        return posterior.clamp_min(_eps);
    }
}
