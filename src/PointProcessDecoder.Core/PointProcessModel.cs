using static TorchSharp.torch;

using PointProcessDecoder.Core.Estimation;
using PointProcessDecoder.Core.Transitions;
using PointProcessDecoder.Core.Encoder;
using PointProcessDecoder.Core.Decoder;
using PointProcessDecoder.Core.StateSpace;
using PointProcessDecoder.Core.Likelihood;
using PointProcessDecoder.Core.Configuration;
using Newtonsoft.Json;

namespace PointProcessDecoder.Core;

public class PointProcessModel : ModelComponent, IModel
{
    private readonly Device _device;
    /// <inheritdoc/>
    public override Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <inheritdoc/>
    public override ScalarType ScalarType => _scalarType;

    private readonly ILikelihood _likelihood;
    public ILikelihood Likelihood => _likelihood;

    private readonly IEncoder _encoderModel;
    public IEncoder Encoder => _encoderModel;

    private readonly IDecoder _decoderModel;
    public IDecoder Decoder => _decoderModel;

    private readonly IStateSpace _stateSpace;
    public IStateSpace StateSpace => _stateSpace;

    private readonly PointProcessModelConfiguration _configuration;

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
        bool ignoreNoSpikes = false,
        double? sigmaRandomWalk = null,
        int? kernelLimit = null,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;

        _stateSpace = stateSpaceType switch
        {
            StateSpaceType.DiscreteUniformStateSpace => new DiscreteUniformStateSpace(
                dimensions: stateSpaceDimensions,
                min: minStateSpace,
                max: maxStateSpace,
                steps: stepsStateSpace,
                device: _device,
                scalarType: _scalarType
            ),
            _ => throw new ArgumentException("Invalid state space type.")
        };

        _encoderModel = encoderType switch
        {
            EncoderType.ClusterlessMarkEncoder => new ClusterlessMarkEncoder(
                estimationMethod: estimationMethod, 
                observationBandwidth: observationBandwidth,
                markDimensions: markDimensions ?? 1, 
                markChannels: markChannels ?? 1,
                markBandwidth: markBandwidth ?? observationBandwidth,
                stateSpace: _stateSpace,
                distanceThreshold: distanceThreshold,
                kernelLimit: kernelLimit,
                device: _device,
                scalarType: _scalarType
            ),
            EncoderType.SortedSpikeEncoder => new SortedSpikeEncoder(
                estimationMethod: estimationMethod, 
                bandwidth: observationBandwidth,
                nUnits: nUnits ?? 1,
                stateSpace: _stateSpace,
                distanceThreshold: distanceThreshold,
                kernelLimit: kernelLimit,
                device: _device,
                scalarType: _scalarType
            ),
            _ => throw new ArgumentException("Invalid encoder type.")
        };

        _decoderModel = decoderType switch
        {
            DecoderType.StateSpaceDecoder => new StateSpaceDecoder(
                transitionsType: transitionsType,
                stateSpace: _stateSpace,
                sigmaRandomWalk: sigmaRandomWalk,
                device: _device,
                scalarType: _scalarType
            ),
            _ => throw new ArgumentException("Invalid decoder type.")
        };

        _likelihood = likelihoodType switch
        {
            LikelihoodType.Poisson => new PoissonLikelihood(
                device: _device,
                scalarType: _scalarType
            ),
            LikelihoodType.Clusterless => new ClusterlessLikelihood(
                ignoreNoSpikes: ignoreNoSpikes,
                device: _device,
                scalarType: _scalarType
            ),
            _ => throw new ArgumentException("Invalid likelihood type.")
        };

        _configuration = new PointProcessModelConfiguration
        {
            EstimationMethod = estimationMethod,
            TransitionsType = transitionsType,
            EncoderType = encoderType,
            DecoderType = decoderType,
            StateSpaceType = stateSpaceType,
            LikelihoodType = likelihoodType,
            MinStateSpace = minStateSpace,
            MaxStateSpace = maxStateSpace,
            StepsStateSpace = stepsStateSpace,
            ObservationBandwidth = observationBandwidth,
            StateSpaceDimensions = stateSpaceDimensions,
            MarkDimensions = markDimensions,
            MarkChannels = markChannels,
            MarkBandwidth = markBandwidth,
            NUnits = nUnits,
            DistanceThreshold = distanceThreshold,
            IgnoreNoSpikes = ignoreNoSpikes,
            SigmaRandomWalk = sigmaRandomWalk,
            ScalarType = scalarType
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
    public override void Save(string basePath)
    {
        JsonSerializer serializer = new()
        {
            Formatting = Formatting.Indented
        };

        Directory.CreateDirectory(basePath);
        
        using StreamWriter sw = new(Path.Combine(basePath, "configuration.json"));
        using JsonWriter writer = new JsonTextWriter(sw);
        serializer.Serialize(writer, _configuration);

        _encoderModel.Save(basePath);
        _decoderModel.Save(basePath);
        _likelihood.Save(basePath);
        _stateSpace.Save(basePath);
    }

    /// <inheritdoc/>
    public new static IModelComponent Load(string basePath)
    {
        // Check that the base path exists
        if (!Directory.Exists(basePath))
        {
            throw new ArgumentException("The base path does not exist.");
        }

        JsonSerializer serializer = new()
        {
            Formatting = Formatting.Indented
        };

        string path = Path.Combine(basePath, "configuration.json");

        if (!File.Exists(path))
        {
            throw new ArgumentException("The configuration file does not exist.");
        }

        using StreamReader sr = new(path);
        using JsonReader reader = new JsonTextReader(sr);
        PointProcessModelConfiguration? configuration = serializer.Deserialize<PointProcessModelConfiguration>(reader) ?? throw new ArgumentException("The configuration file is empty.");
        
        var model = new PointProcessModel(
            estimationMethod: configuration.EstimationMethod,
            transitionsType: configuration.TransitionsType,
            encoderType: configuration.EncoderType,
            decoderType: configuration.DecoderType,
            stateSpaceType: configuration.StateSpaceType,
            likelihoodType: configuration.LikelihoodType,
            minStateSpace: configuration.MinStateSpace,
            maxStateSpace: configuration.MaxStateSpace,
            stepsStateSpace: configuration.StepsStateSpace,
            observationBandwidth: configuration.ObservationBandwidth,
            stateSpaceDimensions: configuration.StateSpaceDimensions,
            markDimensions: configuration.MarkDimensions,
            markChannels: configuration.MarkChannels,
            markBandwidth: configuration.MarkBandwidth,
            nUnits: configuration.NUnits,
            distanceThreshold: configuration.DistanceThreshold,
            ignoreNoSpikes: configuration.IgnoreNoSpikes,
            sigmaRandomWalk: configuration.SigmaRandomWalk,
            scalarType: configuration.ScalarType
        );

        model.Encoder.Load(basePath);
        model.Decoder.Load(basePath);
        model.Likelihood.Load(basePath);
        model.StateSpace.Load(basePath);

        return model;
    }

    /// <inheritdoc/>
    public override void Dispose()
    {
        _encoderModel.Dispose();
        _decoderModel.Dispose();
        _likelihood.Dispose();
        _stateSpace.Dispose();
    }
}
