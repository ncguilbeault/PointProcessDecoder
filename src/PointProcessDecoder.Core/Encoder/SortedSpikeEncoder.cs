using PointProcessDecoder.Core.Estimation;
using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Encoder;

public class SortedSpikeEncoder : IEncoder
{
    private readonly Device _device;
    public Device Device => _device;

    private readonly ScalarType _scalarType;
    public ScalarType ScalarType => _scalarType;

    private Tensor _conditionalIntensities = empty(0);
    private bool _updateConditionalIntensities = true;
    private readonly int _nUnits;
    private readonly IEstimation[] _unitEstimation;
    private readonly IEstimation _observationEstimation;
    private Tensor _spikeCounts = empty(0);
    private Tensor _samples = empty(0);
    private Tensor _rates = empty(0);
    private readonly IStateSpace _stateSpace;
    private readonly double _eps;

    public SortedSpikeEncoder(
        EstimationMethod estimationMethod, 
        double[] bandwidth,
        int nUnits,
        IStateSpace stateSpace,
        double? distanceThreshold = null, 
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        if (nUnits < 1)
        {
            throw new ArgumentException("The number of units must be greater than 0.");
        }

        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _eps = finfo(_scalarType).eps;
        _stateSpace = stateSpace;
        _nUnits = nUnits;
        
        _observationEstimation = GetEstimationMethod(
            estimationMethod, 
            bandwidth, 
            _stateSpace.Dimensions, 
            distanceThreshold
        );

        _unitEstimation = new IEstimation[_nUnits];

        for (int i = 0; i < _nUnits; i++)
        {
            _unitEstimation[i] = GetEstimationMethod(
                estimationMethod, 
                bandwidth, 
                _stateSpace.Dimensions, 
                distanceThreshold,
                device: _device,
                scalarType: _scalarType
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

    /// <summary>
    /// Encode the observations and inputs.
    /// </summary>
    /// <param name="observations"></param>
    /// <param name="inputs"></param>
    /// <exception cref="ArgumentException"></exception>
    public void Encode(Tensor observations, Tensor inputs)
    {
        if (inputs.shape[1] != _nUnits)
        {
            throw new ArgumentException("The number of units in the input tensor must match the expected number of units.");
        }

        if (observations.shape[1] != _stateSpace.Dimensions)
        {
            throw new ArgumentException("The number of observation dimensions must match the dimensions of the state space.");
        }

        _observationEstimation.Fit(observations);

        if (_spikeCounts.numel() == 0)
        {
            _spikeCounts = inputs.to_type(ScalarType.Int32)
                .nan_to_num()
                .sum([0])
                .to_type(_scalarType)
                .to(_device);
            
            _samples = observations.shape[0];
        }
        else
        {
            _spikeCounts += inputs.to_type(ScalarType.Int32)
                .nan_to_num()
                .sum([0])
                .to_type(_scalarType)
                .to(_device);
            _samples += observations.shape[0];
        }

        _rates = _spikeCounts.clamp_min(_eps).log() - _samples.clamp_min(_eps).log();

        for (int i = 0; i < _nUnits; i++)
        {
            _unitEstimation[i].Fit(observations[inputs[TensorIndex.Colon, i]]);
        }

        _updateConditionalIntensities = true;
        _conditionalIntensities = Evaluate().First();
    }

    public IEnumerable<Tensor> Evaluate(params Tensor[] inputs)
    {
        if (_conditionalIntensities.numel() != 0 && !_updateConditionalIntensities)
        {
            return [_conditionalIntensities];
        }
        
        using var _ = NewDisposeScope();
        var observationDensity = _observationEstimation.Evaluate(_stateSpace.Points)
            .clamp_min(_eps)
            .log();
        var conditionalIntensities = new Tensor[_nUnits];

        for (int i = 0; i < _nUnits; i++)
        {
            var unitDensity = _unitEstimation[i].Evaluate(_stateSpace.Points)
                .clamp_min(_eps)
                .log();
            conditionalIntensities[i] = exp(_rates[i] + unitDensity - observationDensity)
                .reshape(_stateSpace.Shape);
        }
        var output = stack(conditionalIntensities, dim: 0)
            .MoveToOuterDisposeScope();
        _updateConditionalIntensities = false;
        return [output];
    }
}
