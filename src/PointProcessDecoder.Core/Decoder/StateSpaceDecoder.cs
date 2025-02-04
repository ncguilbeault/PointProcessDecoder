using static TorchSharp.torch;
using TorchSharp;

using PointProcessDecoder.Core.Transitions;

namespace PointProcessDecoder.Core.Decoder;

public class StateSpaceDecoder : ModelComponent, IDecoder
{
    private readonly Device _device;
    public override Device Device => _device;

    private readonly ScalarType _scalarType;
    public override ScalarType ScalarType => _scalarType;

    public DecoderType DecoderType => DecoderType.StateSpaceDecoder;

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
            _posterior = (_initialState * likelihood[0].flatten())
                .nan_to_num()
                .clamp_min(_eps);
            _posterior /= _posterior.sum();
            output[0] = _posterior.reshape(_stateSpace.Shape);
        }

        for (int i = 1; i < inputs.shape[0]; i++)
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

    // /// <inheritdoc/>
    // public override void Save(string basePath)
    // {
    //     var path = Path.Combine(basePath, "decoder");

    //     if (!Directory.Exists(path))
    //     {
    //         Directory.CreateDirectory(path);
    //     }

    //     _posterior.Save(Path.Combine(path, "posterior.bin"));
    // }

    // /// <inheritdoc/>
    // public override IModelComponent Load(string basePath)
    // {
    //     var path = Path.Combine(basePath, "decoder");

    //     if (!Directory.Exists(path))
    //     {
    //         throw new ArgumentException("The decoder directory does not exist.");
    //     }

    //     _posterior = Tensor.Load(Path.Combine(path, "posterior.bin"));
    //     return this;
    // }

    /// <inheritdoc/>
    public override void Dispose()
    {
        _stateTransitions.Dispose();
        _initialState.Dispose();
        _posterior.Dispose();
    }
}
