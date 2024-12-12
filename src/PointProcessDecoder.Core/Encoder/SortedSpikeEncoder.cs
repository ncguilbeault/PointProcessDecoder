using PointProcessDecoder.Core.Estimation;
using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Encoder;

public class SortedSpikeEncoder : IEncoder
{
    private readonly Device _device;
    public Device Device => _device;

    private readonly ScalarType _scalarType;
    public ScalarType ScalarType => _scalarType;

    private readonly int _observationDimensions;
    public int ObservationDimensions => _observationDimensions;

    private IEnumerable<Tensor>? _conditionalIntensities = null;
    public IEnumerable<Tensor>? ConditionalIntensities => _conditionalIntensities;

    private readonly int _nUnits;
    private readonly IEstimation[] _unitEstimation;
    private readonly Tensor _minLatentSpace;
    private readonly Tensor _maxLatentSpace;
    private readonly Tensor _stepsLatentSpace;

    public SortedSpikeEncoder(
        EstimationMethod estimationMethod, 
        double[] bandwidth, 
        int observationDimensions,
        int nUnits,
        double[] minLatentSpace,
        double[] maxLatentSpace,
        long[] stepsLatentSpace,
        double? distanceThreshold = null, 
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        if (observationDimensions < 1)
        {
            throw new ArgumentException("The number of observation dimensions must be greater than 0.");
        }

        if (nUnits < 1)
        {
            throw new ArgumentException("The number of units must be greater than 0.");
        }

        _observationDimensions = observationDimensions;
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;

        _minLatentSpace = tensor(minLatentSpace, device: _device);
        _maxLatentSpace = tensor(maxLatentSpace, device: _device);
        _stepsLatentSpace = tensor(stepsLatentSpace, device: _device);

        _nUnits = nUnits;

        _unitEstimation = new IEstimation[_nUnits];

        for (int i = 0; i < _nUnits; i++)
        {
            _unitEstimation[i] = GetEstimationMethod(
                estimationMethod, 
                bandwidth, 
                observationDimensions, 
                distanceThreshold
            );
        }
    }

    private IEstimation GetEstimationMethod(EstimationMethod estimationMethod, double[] bandwidth, int dimensions, double? distanceThreshold = null)
    {
        return estimationMethod switch
        {
            EstimationMethod.KernelDensity => new KernelDensity(
                bandwidth, 
                dimensions, 
                device: _device,
                scalarType: _scalarType
            ),
            EstimationMethod.KernelCompression => new KernelCompression(
                bandwidth, 
                dimensions, 
                distanceThreshold, 
                device: _device,
                scalarType: _scalarType
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

        for (int i = 0; i < _nUnits; i++)
        {
            _unitEstimation[i].Fit(observations[inputs[TensorIndex.Colon, i]]);
        }

        _conditionalIntensities = Evaluate();
    }

    public IEnumerable<Tensor> Evaluate()
    {
        return Evaluate(_minLatentSpace, _maxLatentSpace, _stepsLatentSpace);
    }

    public IEnumerable<Tensor> Evaluate(double[] min, double[] max, double[] steps)
    {
        var minLatentSpace = tensor(min, device: _device);
        var maxLatentSpace = tensor(max, device: _device);
        var stepsLatentSpace = tensor(steps, device: _device);

        return Evaluate(minLatentSpace, maxLatentSpace, stepsLatentSpace);
    }

    /// <summary>
    /// Evaluate the density estimation for the given range and steps.
    /// </summary>
    /// <param name="min"></param>
    /// <param name="max"></param>
    /// <param name="steps"></param>
    /// <returns></returns>
    public IEnumerable<Tensor> Evaluate(Tensor min, Tensor max, Tensor steps)
    {
        var unitDensity = new Tensor[_nUnits];

        for (int i = 0; i < _nUnits; i++)
        {
            unitDensity[i] = _unitEstimation[i].Evaluate(min, max, steps);
        }

        return unitDensity;
    }
}
