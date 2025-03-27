using static TorchSharp.torch;

using PointProcessDecoder.Core.Transitions;

namespace PointProcessDecoder.Core.Decoder;

public class HybridStateSpaceReplayClassifier : ModelComponent, IDecoder
{
    private readonly Device _device;
    /// <inheritdoc/>
    public override Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <inheritdoc/>
    public override ScalarType ScalarType => _scalarType;

    public DecoderType DecoderType => DecoderType.HybridStateSpaceReplayClassifier;

    private const int NUM_STATES = 2;

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

    public HybridStateSpaceReplayClassifier(
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

        if (_stayProbability < 0 || _stayProbability > 1)
        {
            throw new ArgumentException("The stay probability must be between 0 and 1.");
        }

        _discreteTransitions = tensor(
            new double[] 
            { 
                _stayProbability, 
                1 - _stayProbability,
                1 - _stayProbability,
                _stayProbability 
            },
            dtype: _scalarType,
            device: _device
        ).reshape(NUM_STATES, NUM_STATES);

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

        var n = _stateSpace.Points.size(0);

        _continuousTransitions = cat([
            uniformTransitions.Transitions.unsqueeze(0),
            uniformTransitions.Transitions.unsqueeze(0),
            uniformTransitions.Transitions.unsqueeze(0),
            randomWalkTransitions.Transitions.unsqueeze(0)
        ], dim: 0).reshape([NUM_STATES, n * NUM_STATES, n]);
        
        _initialState = ones([NUM_STATES, n], dtype: _scalarType, device: _device) / n;
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
            _posterior = (_initialState * likelihood[0])
                .nan_to_num()
                .clamp_min(_eps);
            _posterior /= _posterior.sum();
            output[0] = _posterior.reshape(_posteriorShape);
            startIndex++;
        }

        for (int i = startIndex; i < likelihood.size(0); i++)
        {
            using var discretePred = _discreteTransitions.matmul(_posterior);
            using var discretePredReshaped = discretePred.flatten();
            using var discretePredExpanded = discretePredReshaped.reshape(1, -1, 1);
            using var product = _continuousTransitions * discretePredExpanded;
            using var newPosterior = product.sum(dim: 1);
            _posterior = (newPosterior * likelihood[i])
                .nan_to_num()
                .clamp_min(_eps);
            // _posterior = ((_continuousTransitions * 
            //     _discreteTransitions.matmul(_posterior)
            //         .reshape(1, -1, 1)).sum(dim: 1) * likelihood[i])
            //             .nan_to_num()
            //             .clamp_min(_eps);
            _posterior /= _posterior.sum();
            output[i] = _posterior.reshape(_posteriorShape);
        }
        _posterior.MoveToOuterDisposeScope();
        return output.MoveToOuterDisposeScope();
    }
}
