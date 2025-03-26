using static TorchSharp.torch;

using PointProcessDecoder.Core.Estimation;
using PointProcessDecoder.Core.Transitions;
using PointProcessDecoder.Core.Encoder;
using PointProcessDecoder.Core.Classifier;
using PointProcessDecoder.Core.StateSpace;
using PointProcessDecoder.Core.Likelihood;
using PointProcessDecoder.Core.Configuration;
using Newtonsoft.Json;

namespace PointProcessDecoder.Core;

/// <summary>
/// Represents a replay classifier model.
/// </summary>
public class ReplayClassifierModel
{
    private readonly Device _device;
    /// <inheritdoc/>
    public Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <inheritdoc/>
    public ScalarType ScalarType => _scalarType;

    private readonly ILikelihood _likelihood;
    /// <inheritdoc/>
    public ILikelihood Likelihood => _likelihood;

    private readonly IEncoder _encoderModel;
    /// <inheritdoc/>
    public IEncoder Encoder => _encoderModel;

    private readonly IClassifier _classifierModel;

    private readonly IStateSpace _stateSpace;
    /// <inheritdoc/>
    public IStateSpace StateSpace => _stateSpace;

    public ReplayClassifierModel(
        EstimationMethod estimationMethod,
        EncoderType encoderType,
        ClassifierType classifierType,
        StateSpaceType stateSpaceType,
        LikelihoodType likelihoodType,
        double[] minStateSpace,
        double[] maxStateSpace,
        long[] stepsStateSpace,
        double[] observationBandwidth,
        int stateSpaceDimensions,
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

        _classifierModel = classifierType switch
        {
            ClassifierType.ReplayClassifier => new ReplayClassifier(
                stateSpace: _stateSpace,
                sigmaRandomWalk: sigmaRandomWalk,
                device: _device,
                scalarType: _scalarType
            ),
            _ => throw new ArgumentException("Invalid classifier type.")
        };

        _likelihood = likelihoodType switch
        {
            LikelihoodType.Poisson => new PoissonLikelihood(
                ignoreNoSpikes: ignoreNoSpikes,
                device: _device,
                scalarType: _scalarType
            ),
            _ => throw new ArgumentException("Invalid likelihood type.")
        };
    }

    /// <inheritdoc/>
    public void Encode(Tensor observations, Tensor inputs)
    {
        if (observations.size(1) != _stateSpace.Dimensions)
        {
            throw new ArgumentException("The number of latent dimensions must match the shape of the observations.");
        }
        if (observations.size(0) != inputs.size(0))
        {
            throw new ArgumentException("The number of observations must match the number of inputs.");
        }

        _encoderModel.Encode(observations, inputs);
    }

    /// <inheritdoc/>
    public Tensor Decode(Tensor inputs)
    {
        var conditionalIntensities = _encoderModel.Evaluate(inputs);
        var likelihood = _likelihood.Likelihood(inputs, conditionalIntensities);
        // var posterior = _decoderModel.Decode(likelihood);
        // return new PosteriorData(_stateSpace, posterior);
        return _classifierModel.Decode(likelihood);
    }

    // /// <inheritdoc/>
    // public void Save(string basePath)
    // {
    //     JsonSerializer serializer = new()
    //     {
    //         Formatting = Formatting.Indented
    //     };

    //     Directory.CreateDirectory(basePath);
        
    //     using StreamWriter sw = new(Path.Combine(basePath, "configuration.json"));
    //     using JsonWriter writer = new JsonTextWriter(sw);
    //     serializer.Serialize(writer, _configuration);

    //     _encoderModel.Save(basePath);
    //     _decoderModel.Save(basePath);
    //     _likelihood.Save(basePath);
    //     _stateSpace.Save(basePath);
    // }

    // /// <inheritdoc/>
    // public new static IModelComponent Load(string basePath, Device? device = null)
    // {
    //     // Check that the base path exists
    //     if (!Directory.Exists(basePath))
    //     {
    //         throw new ArgumentException("The base path does not exist.");
    //     }

    //     JsonSerializer serializer = new()
    //     {
    //         Formatting = Formatting.Indented
    //     };

    //     string path = Path.Combine(basePath, "configuration.json");

    //     if (!File.Exists(path))
    //     {
    //         throw new ArgumentException("The configuration file does not exist.");
    //     }

    //     using StreamReader sr = new(path);
    //     using JsonReader reader = new JsonTextReader(sr);
    //     PointProcessModelConfiguration? configuration = serializer.Deserialize<PointProcessModelConfiguration>(reader) ?? throw new ArgumentException("The configuration file is empty.");
        
    //     var model = new PointProcessModel(
    //         estimationMethod: configuration.EstimationMethod,
    //         transitionsType: configuration.TransitionsType,
    //         encoderType: configuration.EncoderType,
    //         decoderType: configuration.DecoderType,
    //         stateSpaceType: configuration.StateSpaceType,
    //         likelihoodType: configuration.LikelihoodType,
    //         minStateSpace: configuration.MinStateSpace,
    //         maxStateSpace: configuration.MaxStateSpace,
    //         stepsStateSpace: configuration.StepsStateSpace,
    //         observationBandwidth: configuration.ObservationBandwidth,
    //         stateSpaceDimensions: configuration.StateSpaceDimensions,
    //         markDimensions: configuration.MarkDimensions,
    //         markChannels: configuration.MarkChannels,
    //         markBandwidth: configuration.MarkBandwidth,
    //         nUnits: configuration.NUnits,
    //         distanceThreshold: configuration.DistanceThreshold,
    //         ignoreNoSpikes: configuration.IgnoreNoSpikes,
    //         sigmaRandomWalk: configuration.SigmaRandomWalk,
    //         kernelLimit: configuration.KernelLimit,
    //         device: device,
    //         scalarType: configuration.ScalarType
    //     );

    //     model.Encoder.Load(basePath);
    //     model.Decoder.Load(basePath);
    //     model.Likelihood.Load(basePath);
    //     model.StateSpace.Load(basePath);

    //     return model;
    // }
}
