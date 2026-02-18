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
    private readonly int _numUnits;
    private readonly IEstimation[] _unitEstimation;
    private readonly IEstimation _covariateEstimation;
    private Tensor _spikeCounts = empty(0);
    private Tensor _samples = empty(0);
    private Tensor _rates = empty(0);
    private readonly IStateSpace _stateSpace;

    /// <summary>
    /// Initializes a new instance of the <see cref="SortedSpikes"/> class.
    /// </summary>
    /// <param name="estimationMethod"></param>
    /// <param name="covariateBandwidth"></param>
    /// <param name="numUnits"></param>
    /// <param name="stateSpace"></param>
    /// <param name="distanceThreshold"></param>
    /// <param name="device"></param>
    /// <param name="scalarType"></param>
    /// <exception cref="ArgumentException"></exception>
    public SortedSpikes(
        EstimationMethod estimationMethod, 
        double[] covariateBandwidth,
        int numUnits,
        IStateSpace stateSpace,
        double? distanceThreshold = null,
        int? kernelLimit = null,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        if (numUnits < 1)
        {
            throw new ArgumentException("The number of units must be greater than 0.", nameof(numUnits));
        }

        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _stateSpace = stateSpace;
        _numUnits = numUnits;
        
        _covariateEstimation = GetEstimationMethod(
            estimationMethod: estimationMethod, 
            bandwidth: covariateBandwidth, 
            dimensions: _stateSpace.Dimensions, 
            distanceThreshold: distanceThreshold,
            kernelLimit: kernelLimit,
            device: device,
            scalarType: scalarType
        );

        _unitEstimation = new IEstimation[_numUnits];

        for (int i = 0; i < _numUnits; i++)
        {
            _unitEstimation[i] = GetEstimationMethod(
                estimationMethod: estimationMethod, 
                bandwidth: covariateBandwidth, 
                dimensions: _stateSpace.Dimensions, 
                distanceThreshold: distanceThreshold,
                kernelLimit: kernelLimit,
                device: device,
                scalarType: scalarType
            );
        }

        _estimations = [_covariateEstimation, .. _unitEstimation];
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
    public void Encode(Tensor covariates, Tensor observations)
    {
        if (covariates.ndim != 2)
        {
            throw new ArgumentException("The covariates tensor must be 2-dimensional with shape (numSamples, covariateDimensions).", nameof(covariates));
        }

        if (observations.ndim != 2)
        {
            throw new ArgumentException("The sorted spikes tensor must be 2-dimensional with shape (numSamples, numUnits).", nameof(observations));
        }

        var covariatesShape = covariates.shape;
        var numCovariateSamples = covariatesShape[0];
        var covariateDimensions = covariatesShape[1];

        var sortedSpikesShape = observations.shape;
        var numSpikeSamples = sortedSpikesShape[0];
        var numUnits = sortedSpikesShape[1];

        if (numUnits != _numUnits)
        {
            throw new ArgumentException("The number of units in the sorted spikes tensor must match the expected number of units.", nameof(observations));
        }

        if (covariateDimensions != _stateSpace.Dimensions)
        {
            throw new ArgumentException("The number of covariate dimensions must match the dimensions of the state space.", nameof(covariates));
        }

        if (numSpikeSamples != numCovariateSamples && numSpikeSamples != 1)
        {
            throw new ArgumentException("The number of samples in the sorted spikes tensor and covariates tensors must match, unless covariates has only one sample.", nameof(covariates));
        }

        _covariateEstimation.Fit(covariates);

        if (_spikeCounts.numel() == 0)
        {
            _spikeCounts = observations.nan_to_num()
                .sum(dim: 0)
                .to(_device);              
            _samples = tensor(numSpikeSamples, device: _device);
        }
        else
        {
            _spikeCounts += observations.nan_to_num()
                .sum(dim: 0)
                .to(_device);
            _samples += numSpikeSamples;
        }

        _rates = _spikeCounts.log() - _samples.log();

        for (int i = 0; i < _numUnits; i++)
        {
            var unitSpikeCounts = _spikeCounts[i].item<long>();

            if (unitSpikeCounts == 0)
            {
                continue;
            }

            if (numCovariateSamples == 1 && numSpikeSamples > 1)
            {
                covariates = covariates.expand(unitSpikeCounts, -1);
                _unitEstimation[i].Fit(covariates);
                continue;
            }

            _unitEstimation[i].Fit(covariates.repeat_interleave(observations[TensorIndex.Colon, i], dim: 0));            
        }

        _updateIntensities = true;
        Evaluate();
    }

    private void EvaluateUnitIntensities()
    {
        using var _ = NewDisposeScope();

        var covariateDensity = _covariateEstimation.Evaluate(_stateSpace.Points)
            .log()
            .nan_to_num();

        _unitIntensities = zeros(
            [_numUnits, _stateSpace.Points.size(0)],
            device: _device,
            dtype: _scalarType
        );

        for (int i = 0; i < _numUnits; i++)
        {
            var unitDensity = _unitEstimation[i].Evaluate(_stateSpace.Points);

            if (unitDensity.numel() == 0)
            {
                continue;
            }

            unitDensity = unitDensity.log()
                .nan_to_num();

            _unitIntensities[i] = _rates[i] + unitDensity - covariateDensity;
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

        var covariateEstimationPath = Path.Combine(path, $"covariateEstimation");

        if (!Directory.Exists(covariateEstimationPath))
        {
            Directory.CreateDirectory(covariateEstimationPath);
        }

        _covariateEstimation.Save(covariateEstimationPath);

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

        var covariateEstimationPath = Path.Combine(path, $"covariateEstimation");

        if (!Directory.Exists(covariateEstimationPath))
        {
            throw new ArgumentException("The covariate estimation directory does not exist within the specified base path.", nameof(basePath));
        }

        _covariateEstimation.Load(covariateEstimationPath);

        for (int i = 0; i < _unitEstimation.Length; i++)
        {
            var unitEstimationPath = Path.Combine(path, $"unitEstimation{i}");

            if (!Directory.Exists(unitEstimationPath))
            {
                throw new ArgumentException("The unit estimation directory does not exist within the specified base path.", nameof(basePath));
            }

            _unitEstimation[i].Load(unitEstimationPath);
        }

        return this;
    }
}
