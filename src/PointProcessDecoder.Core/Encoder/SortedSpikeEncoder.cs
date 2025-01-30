using PointProcessDecoder.Core.Estimation;
using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Encoder;

/// <summary>
/// Represents a sorted spike encoder.
/// </summary>
public class SortedSpikeEncoder : IEncoder
{
    private readonly Device _device;
    /// <inheritdoc/>
    public Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <inheritdoc/>
    public ScalarType ScalarType => _scalarType;

    /// <inheritdoc/>
    public EncoderType EncoderType => EncoderType.SortedSpikeEncoder;

    private Tensor[] _conditionalIntensities = [empty(0)];
    /// <inheritdoc/>
    public Tensor[] ConditionalIntensities => _conditionalIntensities;

    private IEstimation[] _estimations = [];
    /// <inheritdoc/>
    public IEstimation[] Estimations => _estimations;

    private Tensor _unitConditionalIntensities = empty(0);
    private bool _updateConditionalIntensities = true;
    private readonly int _nUnits;
    private readonly IEstimation[] _unitEstimation;
    private readonly IEstimation _observationEstimation;
    private Tensor _spikeCounts = empty(0);
    private Tensor _samples = empty(0);
    private Tensor _rates = empty(0);
    private readonly IStateSpace _stateSpace;
    private readonly double _eps;

    /// <summary>
    /// Initializes a new instance of the <see cref="SortedSpikeEncoder"/> class.
    /// </summary>
    /// <param name="estimationMethod"></param>
    /// <param name="bandwidth"></param>
    /// <param name="nUnits"></param>
    /// <param name="stateSpace"></param>
    /// <param name="distanceThreshold"></param>
    /// <param name="device"></param>
    /// <param name="scalarType"></param>
    /// <exception cref="ArgumentException"></exception>
    public SortedSpikeEncoder(
        EstimationMethod estimationMethod, 
        double[] bandwidth,
        int nUnits,
        IStateSpace stateSpace,
        double? distanceThreshold = null,
        int? kernelLimit = null,
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
            estimationMethod: estimationMethod, 
            bandwidth: bandwidth, 
            dimensions: _stateSpace.Dimensions, 
            distanceThreshold: distanceThreshold,
            kernelLimit: kernelLimit,
            device: device,
            scalarType: scalarType
        );

        _unitEstimation = new IEstimation[_nUnits];

        for (int i = 0; i < _nUnits; i++)
        {
            _unitEstimation[i] = GetEstimationMethod(
                estimationMethod: estimationMethod, 
                bandwidth: bandwidth, 
                dimensions: _stateSpace.Dimensions, 
                distanceThreshold: distanceThreshold,
                kernelLimit: kernelLimit,
                device: device,
                scalarType: scalarType
            );
        }

        _estimations = [_observationEstimation, .. _unitEstimation];
    }

    private static IEstimation GetEstimationMethod(
        EstimationMethod estimationMethod, 
        double[] bandwidth, 
        int dimensions, 
        double? distanceThreshold = null,
        int? kernelLimit = null,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        return estimationMethod switch
        {
            EstimationMethod.KernelDensity => new KernelDensity(
                bandwidth: bandwidth, 
                dimensions: dimensions, 
                device: device,
                scalarType: scalarType
            ),
            EstimationMethod.KernelCompression => new KernelCompression(
                bandwidth: bandwidth, 
                dimensions: dimensions, 
                distanceThreshold: distanceThreshold,
                kernelLimit: kernelLimit,
                device: device,
                scalarType: scalarType
            ),
            _ => throw new ArgumentException("Invalid estimation method.")
        };
    }

    /// <inheritdoc/>
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
            _spikeCounts = inputs.nan_to_num()
                .sum(dim: 0);                
            _samples = observations.shape[0];
        }
        else
        {
            _spikeCounts += inputs.nan_to_num()
                .sum(dim: 0);
            _samples += observations.shape[0];
        }

        _spikeCounts = _spikeCounts
            .to(_device)
            .MoveToOuterDisposeScope();

        _samples = _samples
            .to(_device)
            .MoveToOuterDisposeScope();

        _rates = (_spikeCounts.log() - _samples.log())
            .MoveToOuterDisposeScope();

        var inputMask = inputs.to_type(ScalarType.Bool);

        for (int i = 0; i < _nUnits; i++)
        {
            _unitEstimation[i].Fit(observations[inputMask[TensorIndex.Colon, i]]);
        }

        _updateConditionalIntensities = true;
        _unitConditionalIntensities = Evaluate()
            .First()
            .MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    public IEnumerable<Tensor> Evaluate(params Tensor[] inputs)
    {
        if (_unitConditionalIntensities.numel() != 0 && !_updateConditionalIntensities)
        {
            _conditionalIntensities = [_unitConditionalIntensities];
            return _conditionalIntensities;
        }
        
        using var _ = NewDisposeScope();
        var observationDensity = _observationEstimation.Evaluate(_stateSpace.Points)
            .log();
        var unitConditionalIntensities = new Tensor[_nUnits];

        for (int i = 0; i < _nUnits; i++)
        {
            var unitDensity = _unitEstimation[i].Evaluate(_stateSpace.Points)
                .log();
                
            unitConditionalIntensities[i] = exp(_rates[i] + unitDensity - observationDensity)
                .reshape(_stateSpace.Shape);
        }
        var output = stack(unitConditionalIntensities, dim: 0)
            .MoveToOuterDisposeScope();
        _updateConditionalIntensities = false;
        _conditionalIntensities = [output];
        return _conditionalIntensities;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _observationEstimation.Dispose();
        foreach (var estimation in _unitEstimation)
        {
            estimation.Dispose();
        }
        _estimations = [];
        
        _updateConditionalIntensities = true;
        _conditionalIntensities = [empty(0)];

        _unitConditionalIntensities.Dispose();
        _unitConditionalIntensities = empty(0);

        _spikeCounts.Dispose();
        _spikeCounts = empty(0);

        _samples.Dispose();
        _samples = empty(0);

        _rates.Dispose();
        _rates = empty(0);
    }
}
