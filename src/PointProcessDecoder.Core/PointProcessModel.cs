using static TorchSharp.torch;

using PointProcessDecoder.Core.Estimation;
using PointProcessDecoder.Core.Transitions;
using PointProcessDecoder.Core.Encoder;
using PointProcessDecoder.Core.Decoder;

namespace PointProcessDecoder.Core;

public class PointProcessModel : IModel
{
    private int _latentDimensions;
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

    public PointProcessModel(
        EstimationMethod estimationMethod,
        TransitionsType transitionsType,
        EncoderType encoderType,
        DecoderType decoderType,
        double[] minLatentSpace,
        double[] maxLatentSpace,
        long[] stepsLatentSpace,
        double[] bandwidth,
        int latentDimensions,
        int? markDimensions = null,
        int? markChannels = null,
        int? nUnits = null,
        double? distanceThreshold = null,
        double[]? sigmaLatentSpace = null,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _latentDimensions = latentDimensions;
        _markDimensions = markDimensions ?? 0;
        _markChannels = markChannels ?? 0;
        _nUnits = nUnits ?? 0;

        _encoderModel = encoderType switch
        {
            EncoderType.ClusterlessMarkEncoder => new ClusterlessMarkEncoder(
                estimationMethod, 
                bandwidth, 
                _latentDimensions, 
                _markDimensions, 
                _markChannels,
                minLatentSpace,
                maxLatentSpace,
                stepsLatentSpace,
                distanceThreshold, 
                device: _device
            ),
            EncoderType.SortedSpikeEncoder => new SortedSpikeEncoder(
                estimationMethod, 
                bandwidth, 
                _latentDimensions, 
                _nUnits,
                minLatentSpace,
                maxLatentSpace,
                stepsLatentSpace,
                distanceThreshold, 
                device: _device
            ),
            _ => throw new ArgumentException("Invalid encoder type.")
        };

        _decoderModel = decoderType switch
        {
            DecoderType.SortedSpikeDecoder => new SortedSpikeDecoder(
                transitionsType,
                _latentDimensions,
                minLatentSpace,
                maxLatentSpace,
                stepsLatentSpace,
                sigmaLatentSpace,
                device: _device
            ),
            _ => throw new ArgumentException("Invalid decoder type.")
        };
    }

    /// <summary>
    /// Encodes the observations and data into the latent space.
    /// The observations are in the latent space and are of shape (n, latentDimensions).
    /// The data is in the mark space and is of shape (n, markDimensions, markChannels).
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="data"></param>
    public void Encode(Tensor observations, Tensor inputs)
    {
        if (observations.shape[1] != _latentDimensions)
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
