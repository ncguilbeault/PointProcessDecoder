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

    private readonly IStateTransitions _stateTransitions;
    /// <inheritdoc/>
    public IStateTransitions Transitions => _stateTransitions;

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

        var n = _stateSpace.Points.size(0);
        _initialState = ones(n, dtype: _scalarType, device: _device) / n;
        _posterior = empty(0);
    }

    /// <inheritdoc/>
    public Tensor Decode(Tensor inputs, Tensor likelihood)
    {
        using var _ = NewDisposeScope();

        var outputShape = new long[] { inputs.size(0) }
            .Concat(_stateSpace.Shape)
            .ToArray();

        var output = zeros(outputShape, dtype: _scalarType, device: _device);

        if (_posterior.numel() == 0) {
            _posterior = (_initialState * likelihood[0].flatten())
                .nan_to_num()
                .clamp_min(_eps);
            _posterior /= _posterior.sum();
            output[0] = _posterior.reshape(_stateSpace.Shape);
        }

        for (int i = 1; i < inputs.size(0); i++)
        {
            _posterior = (_stateTransitions.Transitions.matmul(_posterior) * likelihood[i].flatten())
                .nan_to_num()
                .clamp_min(_eps);
            _posterior /= _posterior.sum();
            output[i] = _posterior.reshape(_stateSpace.Shape);
        }
        _posterior.MoveToOuterDisposeScope();
        return output.MoveToOuterDisposeScope();
    }
}
