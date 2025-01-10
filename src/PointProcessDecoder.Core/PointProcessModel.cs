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
    private readonly Device _device;
    /// <inheritdoc/>
    public Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <inheritdoc/>
    public ScalarType ScalarType => _scalarType;

    private readonly LikelihoodType _likelihoodType;
    public LikelihoodType Likelihood => _likelihoodType;
    private readonly ILikelihood _likelihood;

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
        double[] observationBandwidth,
        int stateSpaceDimensions,
        int? markDimensions = null,
        int? markChannels = null,
        double[]? markBandwidth = null,
        int? nUnits = null,
        double? distanceThreshold = null,
        double? sigmaRandomWalk = null,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;

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
                observationBandwidth,
                markDimensions ?? 1, 
                markChannels ?? 1,
                markBandwidth ?? observationBandwidth,
                _stateSpace,
                distanceThreshold, 
                device: _device,
                scalarType: _scalarType
            ),
            EncoderType.SortedSpikeEncoder => new SortedSpikeEncoder(
                estimationMethod, 
                observationBandwidth,
                nUnits ?? 1,
                _stateSpace,
                distanceThreshold, 
                device: _device,
                scalarType: _scalarType
            ),
            _ => throw new ArgumentException("Invalid encoder type.")
        };

        _decoderModel = decoderType switch
        {
            DecoderType.StateSpaceDecoder => new StateSpaceDecoder(
                transitionsType,
                _stateSpace,
                sigmaRandomWalk,
                device: _device,
                scalarType: _scalarType
            ),
            _ => throw new ArgumentException("Invalid decoder type.")
        };

        _likelihoodType = likelihoodType;
        _likelihood = likelihoodType switch
        {
            LikelihoodType.Poisson => new PoissonLikelihood(),
            LikelihoodType.Clusterless => new ClusterlessLikelihood(),
            _ => throw new ArgumentException("Invalid likelihood type.")
        };

    }

    /// <summary>
    /// Encodes the observations and data into the latent space.
    /// The observations are in the latent space and are of shape (n, stateSpaceDimensions).
    /// The inputs are in the neural space.
    /// In the case of sorted units, the inputs are of shape (n, nUnits).
    /// In the case of clusterless marks, the inputs are of shape (n, markDimensions, markChannels).
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="data"></param>
    public void Encode(Tensor observations, Tensor inputs)
    {
        if (observations.shape[1] != _stateSpace.Dimensions)
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
        var conditionalIntensities = _encoderModel.Evaluate(inputs);
        var likelihood = _likelihood.LogLikelihood(inputs, conditionalIntensities);
        return _decoderModel.Decode(inputs, likelihood);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _encoderModel.Dispose();
        _decoderModel.Dispose();
        _stateSpace.Dispose();
    }
}
