using static TorchSharp.torch;

using PointProcessDecoder.Core.Estimation;
using PointProcessDecoder.Core.Transitions;
using PointProcessDecoder.Core.Encoder;

namespace PointProcessDecoder.Core;

public class PointProcessDecoder : PointProcessDecoderBase
{
    private int _latentDimensions;
    /// <inheritdoc/>
    public override int LatentDimensions => _latentDimensions;

    private readonly int _markDimensions;
    private readonly int _markChannels;
    private readonly int _nUnits;

    private readonly Device _device;
    /// <inheritdoc/>
    public override Device Device => _device;

    private Tensor _posterior;
    /// <inheritdoc/>
    public override Tensor Posterior => _posterior;

    private readonly StateTransitions _stateTransitions;
    /// <inheritdoc/>
    public override StateTransitions Transitions => _stateTransitions;

    private readonly EncoderModel _encoderModel;
    public EncoderModel EncoderModel => _encoderModel;

    public PointProcessDecoder(
        EstimationMethod estimationMethod,
        TransitionsType transitionsType,
        EncoderType encoderType,
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
        Device? device = null
    )
    {
        _device = device ?? CPU;
        _latentDimensions = latentDimensions;
        _markDimensions = markDimensions ?? 0;
        _markChannels = markChannels ?? 0;
        _nUnits = nUnits ?? 0;

        _stateTransitions = transitionsType switch
        {
            TransitionsType.Uniform => new UniformTransitions(
                _latentDimensions, 
                minLatentSpace, 
                maxLatentSpace, 
                stepsLatentSpace, 
                device: _device
            ),
            TransitionsType.RandomWalk => new RandomWalkTransitions(
                _latentDimensions, 
                minLatentSpace, 
                maxLatentSpace, 
                stepsLatentSpace, 
                sigmaLatentSpace, 
                device: _device
            ),
            _ => throw new ArgumentException("Invalid transitions type.")
        };

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
            EncoderType.SortedUnitEncoder => new SortedUnitEncoder(
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
        
        var points = _stateTransitions.Points;
        var n = points.shape[0];
        _posterior = ones(n, device: _device) / n;
    }

    /// <summary>
    /// Encode the observations and data into the latent space.
    /// The observations are in the latent space and are of shape (n, latentDimensions).
    /// The data is in the mark space and is of shape (n, markDimensions, markChannels).
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="data"></param>
    public override void Encode(Tensor observations, Tensor data)
    {
        if (observations.shape[1] != _latentDimensions)
        {
            throw new ArgumentException("The number of latent dimensions must match the shape of the observations.");
        }
        if (data.shape[1] != _markDimensions)
        {
            throw new ArgumentException("The number of mark dimensions must match the shape of the data.");
        }
        if (observations.shape[0] != data.shape[0])
        {
            throw new ArgumentException("The number of observations must match the number of data points.");
        }

        _encoderModel.Encode(observations, data);
    }

    public override Tensor Decode(Tensor data)
    {
        throw new NotImplementedException();
    }
}
