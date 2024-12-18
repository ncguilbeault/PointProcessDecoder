using static TorchSharp.torch;

using PointProcessDecoder.Core.Estimation;
using PointProcessDecoder.Core.Transitions;
using PointProcessDecoder.Core.Encoder;
using PointProcessDecoder.Core.Decoder;
using PointProcessDecoder.Core.StateSpace;
using PointProcessDecoder.Core.Likelihood;

namespace PointProcessDecoder.Core;

public class PointProcessModel : IModel
{
    private int _stateSpaceDimensions;
    private readonly int _markDimensions;
    private readonly int _markChannels;
    private readonly int _nUnits;

    private readonly Device _device;
    /// <inheritdoc/>
    public Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <inheritdoc/>
    public ScalarType ScalarType => _scalarType;

    private readonly IEncoder _encoderModel;
    public IEncoder Encoder => _encoderModel;

    private readonly IDecoder _decoderModel;
    public IDecoder Decoder => _decoderModel;

    private readonly IStateSpace _stateSpace;
    public IStateSpace StateSpace => _stateSpace;

    public PointProcessModel(
        EstimationMethod estimationMethod,
        TransitionsType transitionsType,
        EncoderType encoderType,
        DecoderType decoderType,
        StateSpaceType stateSpaceType,
        LikelihoodType likelihoodType,
        double[] minStateSpace,
        double[] maxStateSpace,
        long[] stepsStateSpace,
        double[] bandwidth,
        int stateSpaceDimensions,
        int? markDimensions = null,
        int? markChannels = null,
        int? nUnits = null,
        double? distanceThreshold = null,
        double[]? sigmaRandomWalk = null,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _stateSpaceDimensions = stateSpaceDimensions;
        _markDimensions = markDimensions ?? 0;
        _markChannels = markChannels ?? 0;
        _nUnits = nUnits ?? 0;

        _stateSpace = stateSpaceType switch
        {
            StateSpaceType.DiscreteUniformStateSpace => new DiscreteUniformStateSpace(
                stateSpaceDimensions,
                minStateSpace,
                maxStateSpace,
                stepsStateSpace,
                _device,
                _scalarType
            ),
            _ => throw new ArgumentException("Invalid state space type.")
        };

        _encoderModel = encoderType switch
        {
            EncoderType.ClusterlessMarkEncoder => new ClusterlessMarkEncoder(
                estimationMethod, 
                bandwidth,
                _markDimensions, 
                _markChannels,
                _stateSpace,
                distanceThreshold, 
                device: _device
            ),
            EncoderType.SortedSpikeEncoder => new SortedSpikeEncoder(
                estimationMethod, 
                bandwidth,
                _nUnits,
                _stateSpace,
                distanceThreshold, 
                device: _device
            ),
            _ => throw new ArgumentException("Invalid encoder type.")
        };

        _decoderModel = decoderType switch
        {
            DecoderType.StateSpaceDecoder => new StateSpaceDecoder(
                transitionsType,
                likelihoodType,
                _stateSpace,
                sigmaRandomWalk,
                device: _device
            ),
            _ => throw new ArgumentException("Invalid decoder type.")
        };
    }

    /// <summary>
    /// Encodes the observations and data into the latent space.
    /// The observations are in the latent space and are of shape (n, stateSpaceDimensions).
    /// The data is in the mark space and is of shape (n, markDimensions, markChannels).
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="data"></param>
    public void Encode(Tensor observations, Tensor inputs)
    {
        if (observations.shape[1] != _stateSpaceDimensions)
        {
            throw new ArgumentException("The number of latent dimensions must match the shape of the observations.");
        }
        if (observations.shape[0] != inputs.shape[0])
        {
            throw new ArgumentException("The number of observations must match the number of inputs.");
        }

        _encoderModel.Encode(observations, inputs);
    }

    /// <summary>
    /// Decodes the inputs into the latent space.
    /// </summary>
    /// <param name="data"></param>
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public Tensor Decode(Tensor inputs)
    {
        if (_encoderModel.ConditionalIntensities == null)
        {
            throw new InvalidOperationException("The encoder must be run before decoding.");
        }

        return _decoderModel.Decode(inputs, _encoderModel.ConditionalIntensities);
    }
}
