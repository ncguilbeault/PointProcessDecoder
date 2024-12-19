using PointProcessDecoder.Core.Estimation;
using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Encoder;

public class ClusterlessMarkEncoder : IEncoder
{
    private readonly Device _device;
    public Device Device => _device;

    private readonly ScalarType _scalarType;
    public ScalarType ScalarType => _scalarType;

    private readonly IEstimation _observationEstimation;
    private readonly IEstimation[] _channelEstimation;
    private readonly IEstimation[] _markEstimation;

    private readonly IStateSpace _stateSpace;
    private bool _updateConditionalIntensities = true;

    private Tensor _markConditionalIntensities = empty(0);
    private Tensor[] _channelEstimates = [];
    private Tensor _channelConditionalIntensities = empty(0);
    private Tensor _observationDensity = empty(0);

    private Tensor _spikeCounts = empty(0);
    private Tensor _samples = empty(0);
    private Tensor _rates = empty(0);
    private readonly int _markDimensions;
    private int _markChannels;
    private readonly double _eps;

    public ClusterlessMarkEncoder(
        EstimationMethod estimationMethod, 
        double[] observationBandwidth, 
        int markDimensions,
        int markChannels,
        double[] markBandwidth,
        IStateSpace stateSpace,
        double? distanceThreshold = null, 
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        if (markChannels < 1)
        {
            throw new ArgumentException("The number of mark channels must be greater than 0.");
        }

        if (markDimensions < 1)
        {
            throw new ArgumentException("The number of mark dimensions must be greater than 0.");
        }
        
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _eps = finfo(_scalarType).eps;
        _markDimensions = markDimensions;
        _markChannels = markChannels;
        _stateSpace = stateSpace;

        _observationEstimation = GetEstimationMethod(
            estimationMethod, 
            observationBandwidth, 
            _stateSpace.Dimensions, 
            distanceThreshold
        );

        _channelEstimation = new IEstimation[_markChannels];
        _markEstimation = new IEstimation[_markChannels];

        for (int i = 0; i < _markChannels; i++)
        {
            _channelEstimation[i] = GetEstimationMethod(
                estimationMethod, 
                observationBandwidth, 
                _stateSpace.Dimensions, 
                distanceThreshold
            );

            _markEstimation[i] = GetEstimationMethod(
                estimationMethod, 
                markBandwidth,
                markDimensions, 
                distanceThreshold
            );
        }
    }

    private static IEstimation GetEstimationMethod(
        EstimationMethod estimationMethod, 
        double[] bandwidth, 
        int dimensions, 
        double? distanceThreshold = null,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        return estimationMethod switch
        {
            EstimationMethod.KernelDensity => new KernelDensity(
                bandwidth, 
                dimensions, 
                device: device,
                scalarType: scalarType
            ),
            EstimationMethod.KernelCompression => new KernelCompression(
                bandwidth, 
                dimensions, 
                distanceThreshold,
                device: device,
                scalarType: scalarType
            ),
            _ => throw new ArgumentException("Invalid estimation method.")
        };
    }

    // mark is a tensor of shape (nSamples, markDimensions, markChannels)
    // observation is a tensor of shape (nSamples, observationDimensions)
    public void Encode(Tensor observations, Tensor marks)
    {
        if (marks.shape[1] != _markDimensions)
        {
            throw new ArgumentException("The number of mark dimensions must match the shape of the marks tensor on dimension 1.");
        }

        if (marks.shape[2] != _markChannels)
        {
            throw new ArgumentException("The number of mark channels must match the shape of the marks tensor on dimension 2.");
        }

        if (observations.shape[1] != _stateSpace.Dimensions)
        {
            throw new ArgumentException("The number of observation dimensions must match the dimensions of the state space.");
        }

        _observationEstimation.Fit(observations);

        if (_spikeCounts.numel() == 0)
        {
            _spikeCounts = (marks.sum(dim: 1) > 0).sum(dim: 0);
            _samples = observations.shape[0];
        }
        else
        {
            _spikeCounts += (marks.sum(dim: 1) > 0).sum(dim: 0);
            _samples += observations.shape[0];
        }

        _spikeCounts = _spikeCounts
            .to(_device)
            .MoveToOuterDisposeScope();

        _samples = _samples
            .to(_device)
            .MoveToOuterDisposeScope();

        _rates = (_spikeCounts.clamp_min(_eps).log() - _samples.clamp_min(_eps).log())
            .MoveToOuterDisposeScope();
        
        var mask = marks.sum(dim: 1) > 0;

        for (int i = 0; i < _markChannels; i++)
        {
            var channelObservation = observations[mask[TensorIndex.Colon, i]];
            var markObservation = marks[TensorIndex.Tensor(mask[TensorIndex.Colon, i]), TensorIndex.Colon, i];
            _channelEstimation[i].Fit(channelObservation);
            _markEstimation[i].Fit(markObservation);
        }

        _updateConditionalIntensities = true;
        _channelConditionalIntensities = Evaluate()
            .First()
            .MoveToOuterDisposeScope();
    }

    private Tensor EvaluateMarkConditionalIntensities(Tensor inputs)
    {
        using var _ = NewDisposeScope();

        var markConditionalIntensities = new Tensor[_markChannels];

        for (int i = 0; i < _markChannels; i++)
        {
            var marks = inputs[TensorIndex.Ellipsis, i];
            var markEstimate = _markEstimation[i].Estimate(marks);
            var markDensity = markEstimate.matmul(_channelEstimates[i].T)
                .clamp_min(_eps)
                .log();

            var jointDensity = markDensity + _channelConditionalIntensities[i].unsqueeze(0);

            markConditionalIntensities[i] = jointDensity
                .MoveToOuterDisposeScope();
            // markConditionalIntensities[i] = (jointDensity - _observationDensity.unsqueeze(0));

        }
        return stack(markConditionalIntensities, dim: 0)
            .MoveToOuterDisposeScope();
    }

    private Tensor EvaluateChannelConditionalIntensities()
    {
        using var _ = NewDisposeScope();

        _observationDensity = _observationEstimation.Evaluate(_stateSpace.Points)
            .clamp_min(_eps)
            .log();

        var channelConditionalIntensities = new Tensor[_markChannels];
        _channelEstimates = new Tensor[_markChannels];

        for (int i = 0; i < _markChannels; i++)
        {
            _channelEstimates[i] = _channelEstimation[i].Estimate(_stateSpace.Points)
                .MoveToOuterDisposeScope();

            var channelDensity = _channelEstimates[i]
                .mean(dimensions: [1])
                .clamp_min(_eps)
                .log()
                .MoveToOuterDisposeScope();

            channelConditionalIntensities[i] = (_rates[i] + channelDensity - _observationDensity)
                .clamp_min(_eps);
        }

        _observationDensity.MoveToOuterDisposeScope();
        _updateConditionalIntensities = false;

        return stack(channelConditionalIntensities, dim: 0)
            .MoveToOuterDisposeScope();
    }

    public IEnumerable<Tensor> Evaluate(params Tensor[] inputs)
    {
        if (_updateConditionalIntensities)
        {
            _channelConditionalIntensities = EvaluateChannelConditionalIntensities()
                .MoveToOuterDisposeScope();
        }

        if (inputs.Length > 0)
        {
            _markConditionalIntensities = EvaluateMarkConditionalIntensities(inputs[0])
                .MoveToOuterDisposeScope();
        }

        return [_channelConditionalIntensities, _markConditionalIntensities];
    }
}
