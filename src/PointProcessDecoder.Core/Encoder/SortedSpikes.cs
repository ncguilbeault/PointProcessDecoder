using PointProcessDecoder.Core.Estimation;
using TorchSharp;
using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Encoder;

/// <summary>
/// Represents a sorted spike encoder.
/// </summary>
public class SortedSpikes : ModelComponent, IEncoder
{
    private readonly Device _device;
    /// <inheritdoc/>
    public override Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <inheritdoc/>
    public override ScalarType ScalarType => _scalarType;

    /// <inheritdoc/>
    public EncoderType EncoderType => EncoderType.SortedSpikes;

    /// <inheritdoc/>
    public Tensor[] Intensities => [_unitIntensities];

    private IEstimation[] _estimations = [];
    /// <inheritdoc/>
    public IEstimation[] Estimations => _estimations;

    private Tensor _unitIntensities = empty(0);
    private bool _updateIntensities = true;
    private readonly int _nUnits;
    private readonly IEstimation[] _unitEstimation;
    private readonly IEstimation _observationEstimation;
    private Tensor _spikeCounts = empty(0);
    private Tensor _samples = empty(0);
    private Tensor _rates = empty(0);
    private readonly IStateSpace _stateSpace;

    /// <summary>
    /// Initializes a new instance of the <see cref="SortedSpikes"/> class.
    /// </summary>
    /// <param name="estimationMethod"></param>
    /// <param name="bandwidth"></param>
    /// <param name="nUnits"></param>
    /// <param name="stateSpace"></param>
    /// <param name="distanceThreshold"></param>
    /// <param name="device"></param>
    /// <param name="scalarType"></param>
    /// <exception cref="ArgumentException"></exception>
    public SortedSpikes(
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
                kernelLimit: kernelLimit,
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
        if (inputs.size(1) != _nUnits)
        {
            throw new ArgumentException("The number of units in the input tensor must match the expected number of units.");
        }

        if (observations.size(1) != _stateSpace.Dimensions)
        {
            throw new ArgumentException("The number of observation dimensions must match the dimensions of the state space.");
        }

        _observationEstimation.Fit(observations);

        if (_spikeCounts.numel() == 0)
        {
            _spikeCounts = inputs.nan_to_num()
                .sum(dim: 0)
                .to(_device);              
            _samples = tensor(observations.size(0), device: _device);
        }
        else
        {
            _spikeCounts += inputs.nan_to_num()
                .sum(dim: 0)
                .to(_device);
            _samples += observations.size(0);
        }

        _rates = _spikeCounts.log() - _samples.log();

        for (int i = 0; i < _nUnits; i++)
        {
            _unitEstimation[i].Fit(observations.repeat_interleave(inputs[TensorIndex.Colon, i], dim: 0));
        }

        _updateIntensities = true;
        Evaluate();
    }

    private void EvaluateUnitIntensities()
    {
        using var _ = NewDisposeScope();

        var observationDensity = _observationEstimation.Evaluate(_stateSpace.Points)
            .log()
            .nan_to_num();

        _unitIntensities = zeros(
            [_nUnits, _stateSpace.Points.size(0)],
            device: _device,
            dtype: _scalarType
        );

        for (int i = 0; i < _nUnits; i++)
        {
            var unitDensity = _unitEstimation[i].Evaluate(_stateSpace.Points);

            if (unitDensity.numel() == 0)
            {
                continue;
            }

            unitDensity = unitDensity.log()
                .nan_to_num();

            _unitIntensities[i] = _rates[i] + unitDensity - observationDensity;
        }

        _unitIntensities.MoveToOuterDisposeScope();

        _updateIntensities = false;
    }

    /// <inheritdoc/>
    public IEnumerable<Tensor> Evaluate(params Tensor[] inputs)
    {
        if (_unitIntensities.numel() == 0 || _updateIntensities)
        {
            EvaluateUnitIntensities();
        }
        
        return Intensities;
    }

    /// <inheritdoc/>
    public override void Save(string basePath)
    {
        var path = Path.Combine(basePath, "encoder");

        if (!Directory.Exists(path))
        {
            Directory.CreateDirectory(path);
        }

        _spikeCounts.Save(Path.Combine(path, "spikeCounts.bin"));
        _samples.Save(Path.Combine(path, "samples.bin"));
        _rates.Save(Path.Combine(path, "rates.bin"));
        _unitIntensities.Save(Path.Combine(path, "unitIntensities.bin"));

        var observationEstimationPath = Path.Combine(path, $"observationEstimation");

        if (!Directory.Exists(observationEstimationPath))
        {
            Directory.CreateDirectory(observationEstimationPath);
        }

        _observationEstimation.Save(observationEstimationPath);

        for (int i = 0; i < _unitEstimation.Length; i++)
        {
            var unitEstimationPath = Path.Combine(path, $"unitEstimation{i}");

            if (!Directory.Exists(unitEstimationPath))
            {
                Directory.CreateDirectory(unitEstimationPath);
            }
            
            _unitEstimation[i].Save(unitEstimationPath);
        }
    }

    /// <inheritdoc/>
    public override IModelComponent Load(string basePath)
    {
        var path = Path.Combine(basePath, "encoder");

        if (!Directory.Exists(path))
        {
            throw new ArgumentException("The encoder directory does not exist.");
        }

        _spikeCounts = Tensor.Load(Path.Combine(path, "spikeCounts.bin")).to(_device);
        _samples = Tensor.Load(Path.Combine(path, "samples.bin")).to(_device);
        _rates = Tensor.Load(Path.Combine(path, "rates.bin")).to(_device);
        _unitIntensities = Tensor.Load(Path.Combine(path, "unitIntensities.bin")).to(_device);

        var observationEstimationPath = Path.Combine(path, $"observationEstimation");

        if (!Directory.Exists(observationEstimationPath))
        {
            throw new ArgumentException("The observation estimation directory does not exist.");
        }

        _observationEstimation.Load(observationEstimationPath);

        for (int i = 0; i < _unitEstimation.Length; i++)
        {
            var unitEstimationPath = Path.Combine(path, $"unitEstimation{i}");

            if (!Directory.Exists(unitEstimationPath))
            {
                throw new ArgumentException("The unit estimation directory does not exist.");
            }

            _unitEstimation[i].Load(unitEstimationPath);
        }

        return this;
    }
}
