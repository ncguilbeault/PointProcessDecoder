using PointProcessDecoder.Core.Estimation;
using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Encoder;

public class ClusterlessMarkEncoder : IEncoder
{
    private readonly Device _device;
    public Device Device => _device;

    private readonly ScalarType _scalarType;
    public ScalarType ScalarType => _scalarType;

    private IEnumerable<Tensor>? _conditionalIntensities = null;
    public IEnumerable<Tensor>? ConditionalIntensities => _conditionalIntensities;

    private readonly IEstimation _observationEstimation;
    private readonly IEstimation[] _observationAtMarkEstimation;
    private readonly IEstimation[] _jointEstimation;

    private readonly IStateSpace _stateSpace;
    private Tensor _rates = empty(0);
    private Tensor _samples = empty(0);
    private readonly int _markDimensions;
    private int _markChannels;

    public ClusterlessMarkEncoder(
        EstimationMethod estimationMethod, 
        double[] bandwidth, 
        int markDimensions,
        int markChannels,
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
        _markDimensions = markDimensions;
        _markChannels = markChannels;
        _stateSpace = stateSpace;

        _observationEstimation = GetEstimationMethod(
            estimationMethod, 
            bandwidth, 
            _stateSpace.Dimensions, 
            distanceThreshold
        );

        _observationAtMarkEstimation = new IEstimation[_markChannels];
        _jointEstimation = new IEstimation[_markChannels];

        for (int i = 0; i < _markChannels; i++)
        {
            _observationAtMarkEstimation[i] = GetEstimationMethod(
                estimationMethod, 
                bandwidth, 
                _stateSpace.Dimensions, 
                distanceThreshold
            );

            _jointEstimation[i] = GetEstimationMethod(
                estimationMethod, 
                bandwidth,
                _stateSpace.Dimensions + markDimensions, 
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

        // if (_rates.numel() == 0)
        // {
        //     _rates = inputs.to_type(ScalarType.Int32)
        //         .sum([0], type: _scalarType)
        //         .log()
        //         .nan_to_num()
        //         .to_type(_scalarType)
        //         .to(_device);
            
        //     _samples = observations.shape[0];
        // }
        // else
        // {
        //     var totalSamples = _samples + observations.shape[0];
        //     var oldRates = _rates * (_samples / totalSamples);
        //     var newRates = inputs.to_type(ScalarType.Int32)
        //         .sum([0], type: _scalarType)
        //         .log()
        //         .nan_to_num()
        //         .to_type(_scalarType)
        //         .to(_device) * (observations.shape[0] / totalSamples);
        //     _rates = oldRates + newRates;
        //     _samples = totalSamples;
        // }

        var mask = marks.sum(dim: 1) > 0;

        for (int i = 0; i < _markChannels; i++)
        {
            var observationAtMark = observations[mask[TensorIndex.Colon, i]];
            var markAtMark = marks[mask[TensorIndex.Colon, i], TensorIndex.Colon, i];
            _observationAtMarkEstimation[i].Fit(observationAtMark);
            _jointEstimation[i].Fit(cat([observationAtMark, markAtMark], dim: 1));
        }

        _conditionalIntensities = Evaluate();
    }

    public IEnumerable<Tensor> Evaluate()
    {
        var observationDensity = _observationEstimation.Evaluate(_stateSpace.Points);
        var observationAtMarkDensity = new Tensor[_markChannels];
        var jointDensity = new Tensor[_markChannels];

        for (int i = 0; i < _markChannels; i++)
        {
            observationAtMarkDensity[i] = _observationAtMarkEstimation[i].Evaluate(_stateSpace.Points);
            jointDensity[i] = _jointEstimation[i].Evaluate(_stateSpace.Points);
        }

        return [observationDensity, concat(observationAtMarkDensity), concat(jointDensity)];
    }
}
