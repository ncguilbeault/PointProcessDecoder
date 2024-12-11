using PointProcessDecoder.Core.Estimation;
using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Encoder;

public class ClusterlessMarkEncoder : EncoderModel
{
    private readonly Device _device;
    public override Device Device => _device;

    private readonly int _observationDimensions;
    public int ObservationDimensions => _observationDimensions;

    private readonly int _markDimensions;
    public int MarkDimensions => _markDimensions;

    private int _markChannels;
    public int MarkChannels => _markChannels;

    private readonly DensityEstimation _observationEstimation;
    private readonly DensityEstimation[] _observationAtMarkEstimation;
    private readonly DensityEstimation[] _jointEstimation;

    private readonly Tensor _minLatentSpace;
    private readonly Tensor _maxLatentSpace;
    private readonly Tensor _stepsLatentSpace;

    public ClusterlessMarkEncoder(
        EstimationMethod estimationMethod, 
        double[] bandwidth, 
        int observationDimensions, 
        int markDimensions,
        int markChannels,
        double[] minLatentSpace,
        double[] maxLatentSpace,
        long[] stepsLatentSpace,
        double? distanceThreshold = null, 
        Device? device = null
    )
    {
        if (observationDimensions < 1)
        {
            throw new ArgumentException("The number of observation dimensions must be greater than 0.");
        }

        if (markChannels < 1)
        {
            throw new ArgumentException("The number of mark channels must be greater than 0.");
        }

        if (markDimensions < 1)
        {
            throw new ArgumentException("The number of mark dimensions must be greater than 0.");
        }

        if (minLatentSpace.Length != observationDimensions || maxLatentSpace.Length != observationDimensions || stepsLatentSpace.Length != observationDimensions)
        {
            throw new ArgumentException("The length of minLatentSpace, maxLatentSpace, and stepsLatentSpace must match the number of observation dimensions.");
        }

        _observationDimensions = observationDimensions;
        _markDimensions = markDimensions;
        _device = device ?? CPU;
        _markChannels = markChannels;

        _minLatentSpace = tensor(minLatentSpace, device: _device);
        _maxLatentSpace = tensor(maxLatentSpace, device: _device);
        _stepsLatentSpace = tensor(stepsLatentSpace, device: _device);

        _observationEstimation = GetEstimationMethod(estimationMethod, bandwidth, observationDimensions, distanceThreshold);

        _observationAtMarkEstimation = new DensityEstimation[_markChannels];
        _jointEstimation = new DensityEstimation[_markChannels];

        for (int i = 0; i < _markChannels; i++)
        {
            _observationAtMarkEstimation[i] = GetEstimationMethod(estimationMethod, bandwidth, observationDimensions, distanceThreshold);
            _jointEstimation[i] = GetEstimationMethod(estimationMethod, bandwidth, observationDimensions + markDimensions, distanceThreshold);
        }
    }

    private DensityEstimation GetEstimationMethod(EstimationMethod estimationMethod, double[] bandwidth, int dimensions, double? distanceThreshold = null)
    {
        return estimationMethod switch
        {
            EstimationMethod.KernelDensity => new KernelDensity(bandwidth, dimensions, device: _device),
            EstimationMethod.KernelCompression => new KernelCompression(bandwidth, dimensions, distanceThreshold, device: _device),
            _ => throw new ArgumentException("Invalid estimation method.")
        };
    }

    // mark is a tensor of shape (nSamples, markDimensions, markChannels)
    // observation is a tensor of shape (nSamples, observationDimensions)
    public override void Encode(Tensor observations, Tensor marks)
    {
        observations.to(_device);
        marks.to(_device);
        
        _observationEstimation.Fit(observations);

        // mark is a tensor of shape (nSamples, markDimensions, markChannels)
        // need to calculate the mask where mark is present for each channel
        // mask should be calculated in one go
        var mask = marks.sum(dim: 1) > 0; // mask is a tensor of shape (nSamples, markChannels)

        // for each channel, fit the observations where a mark was present
        for (int i = 0; i < _markChannels; i++)
        {
            var observationAtMark = observations[mask[TensorIndex.Colon, i]];
            var markAtMark = marks[mask[TensorIndex.Colon, i], TensorIndex.Colon, i];
            _observationAtMarkEstimation[i].Fit(observationAtMark);
            _jointEstimation[i].Fit(cat([observationAtMark, markAtMark], dim: 1));
        }
    }

    public override IEnumerable<Tensor> Evaluate()
    {
        return Evaluate(_minLatentSpace, _maxLatentSpace, _stepsLatentSpace);
    }

    public IEnumerable<Tensor> Evaluate(double[] min, double[] max, double[] steps)
    {
        if (_observationDimensions != min.Length || _observationDimensions != max.Length || _observationDimensions != steps.Length)
        {
            throw new ArgumentException("The length of min, max, and steps must match the number of observation dimensions");
        }
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
    /// <returns>
    /// A list of tensors containing the density estimation for the observations, the density estimation for the observations at each mark channel, and the joint density of the observations and marks.
    /// </returns>
    public override IEnumerable<Tensor> Evaluate(Tensor min, Tensor max, Tensor steps)
    {
        var observationDensity = _observationEstimation.Evaluate(min, max, steps);
        var observationAtMarkDensity = new Tensor[_markChannels];
        var jointDensity = new Tensor[_markChannels];

        for (int i = 0; i < _markChannels; i++)
        {
            observationAtMarkDensity[i] = _observationAtMarkEstimation[i].Evaluate(min, max, steps);
            jointDensity[i] = _jointEstimation[i].Evaluate(min, max, steps);
        }

        return [observationDensity, concat(observationAtMarkDensity), concat(jointDensity)];
    }
}
