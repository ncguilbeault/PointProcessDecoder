using static TorchSharp.torch;

using PointProcessDecoder.Core.Transitions;

namespace PointProcessDecoder.Core.Decoder;

public class StateSpaceDecoder : ModelComponent, IDecoder
{
    private readonly Device _device;
    /// <inheritdoc/>
    public override Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <inheritdoc/>
    public override ScalarType ScalarType => _scalarType;

    public DecoderType DecoderType => DecoderType.StateSpaceDecoder;

    private readonly Tensor _initialState = empty(0);
    /// <inheritdoc/>
    public Tensor InitialState => _initialState;

    private readonly Tensor _stateTransitions;
    /// <inheritdoc/>
    public Tensor[] Transitions => [_stateTransitions];

    private Tensor _posterior;
    /// <summary>
    /// The posterior distribution over the latent space.
    /// </summary>
    public Tensor Posterior => _posterior;

    private readonly IStateSpace _stateSpace;
    private readonly double _eps;

    /// <summary>
    /// Initializes a new instance of the <see cref="StateSpaceDecoder"/> class.
    /// </summary>
    /// <param name="transitionsType"></param>
    /// <param name="stateSpace"></param>
    /// <param name="sigmaRandomWalk"></param>
    /// <param name="device"></param>
    /// <param name="scalarType"></param>
    /// <exception cref="ArgumentException"></exception>
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
            TransitionsType.Uniform => new Uniform(
                _stateSpace,
                device: _device,
                scalarType: _scalarType
            ).Transitions,
            TransitionsType.RandomWalk => new RandomWalk(
                _stateSpace,
                sigmaRandomWalk, 
                device: _device,
                scalarType: _scalarType
            ).Transitions,
            TransitionsType.Stationary => new Stationary(
                _stateSpace,
                device: _device,
                scalarType: _scalarType
            ).Transitions,
            TransitionsType.ReciprocalGaussian => new ReciprocalGaussian(
                _stateSpace,
                device: _device,
                scalarType: _scalarType
            ).Transitions,
            _ => throw new ArgumentException("Invalid transitions type.")
        };

        var n = _stateSpace.Points.size(0);
        _initialState = ones(n, dtype: _scalarType, device: _device) / n;
        _posterior = empty(0);
    }

    /// <inheritdoc/>
    public Tensor Decode(Tensor likelihood)
    {
        using var _ = NewDisposeScope();

        var outputShape = new long[] { likelihood.size(0) }
            .Concat(_stateSpace.Shape)
            .ToArray();

        var output = zeros(outputShape, dtype: _scalarType, device: _device);

        var startIndex = 0;

        if (_posterior.numel() == 0) {
            _posterior = UpdatePosterior(_initialState, likelihood[0]);
            output[0] = _posterior.reshape(_stateSpace.Shape);
            startIndex++;
        }

        for (int i = startIndex; i < likelihood.size(0); i++)
        {
            using var prediction = _stateTransitions.matmul(_posterior);
            _posterior = UpdatePosterior(prediction, likelihood[i]);
            output[i] = _posterior.reshape(_stateSpace.Shape);
        }
        
        _posterior.MoveToOuterDisposeScope();
        return output.MoveToOuterDisposeScope();
    }

    private Tensor UpdatePosterior(Tensor prior, Tensor likelihood)
    {
        var posterior = prior * likelihood;

        if (posterior.nansum().item<float>() == 0)
        {
            posterior = _initialState * likelihood;
        }

        posterior /= posterior.nansum();

        return posterior
            .nan_to_num()
            .clamp_min(_eps);
    }
}
