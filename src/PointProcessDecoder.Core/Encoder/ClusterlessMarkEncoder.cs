using PointProcessDecoder.Core.Estimation;
using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Encoder;

/// <summary>
/// Represents a clusterless mark encoder.
/// </summary>
public class ClusterlessMarkEncoder : IEncoder
{
    private readonly Device _device;
    /// <inheritdoc/>
    public Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <inheritdoc/>
    public ScalarType ScalarType => _scalarType;

    /// <inheritdoc/>
    public EncoderType EncoderType => EncoderType.ClusterlessMarkEncoder;

    private Tensor[] _conditionalIntensities = [empty(0)];
    /// <inheritdoc/>
    public Tensor[] ConditionalIntensities => _conditionalIntensities;

    private IEstimation[] _estimations = [];
    /// <inheritdoc/>
    public IEstimation[] Estimations => _estimations;

    private readonly IEstimation _observationEstimation;
    private readonly IEstimation[] _channelEstimation;
    private readonly IEstimation[] _markEstimation;

    private readonly IStateSpace _stateSpace;
    private bool _updateConditionalIntensities = true;

    private Tensor[] _markStateSpaceKernelEstimates = [];
    private Tensor _markConditionalIntensities = empty(0);
    private Tensor[] _channelEstimates = [];
    private Tensor _channelConditionalIntensities = empty(0);
    private Tensor _observationDensity = empty(0);

    private Tensor _spikeCounts = empty(0);
    private Tensor _samples = empty(0);
    private Tensor _rates = empty(0);
    private readonly int _markDimensions;
    private int _markChannels;
    private readonly Action<int, Tensor, Tensor> _markFitMethod;
    private readonly Func<int, Tensor, Tensor> _estimateMarkConditionalIntensityMethod;
    private readonly Func<IEstimation, Tensor> _estimateMarkStateSpaceKernelMethod;

    /// <summary>
    /// Initializes a new instance of the <see cref="ClusterlessMarkEncoder"/> class.
    /// </summary>
    /// <param name="estimationMethod"></param>
    /// <param name="observationBandwidth"></param>
    /// <param name="markDimensions"></param>
    /// <param name="markChannels"></param>
    /// <param name="markBandwidth"></param>
    /// <param name="stateSpace"></param>
    /// <param name="distanceThreshold"></param>
    /// <param name="device"></param>
    /// <param name="scalarType"></param>
    /// <exception cref="ArgumentException"></exception>
    public ClusterlessMarkEncoder(
        EstimationMethod estimationMethod, 
        double[] observationBandwidth, 
        int markDimensions,
        int markChannels,
        double[] markBandwidth,
        IStateSpace stateSpace,
        double? distanceThreshold = null,
        int? kernelLimit = null,
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

        _channelEstimation = new IEstimation[_markChannels];
        _markEstimation = new IEstimation[_markChannels];

        switch (estimationMethod)
        {
            case EstimationMethod.KernelDensity:

                _observationEstimation = new KernelDensity(
                    bandwidth: observationBandwidth, 
                    dimensions: _stateSpace.Dimensions, 
                    device: device,
                    scalarType: scalarType
                );

                for (int i = 0; i < _markChannels; i++)
                {
                    _channelEstimation[i] = new KernelDensity(
                        bandwidth: observationBandwidth, 
                        dimensions: _stateSpace.Dimensions, 
                        device: device,
                        scalarType: scalarType
                    );

                    _markEstimation[i] = new KernelDensity(
                        bandwidth: markBandwidth, 
                        dimensions: _markDimensions, 
                        device: device,
                        scalarType: scalarType
                    );
                }

                _markFitMethod = FitMarksFactoredMethod;
                _estimateMarkConditionalIntensityMethod = EstimateMarksFactoredMethod;
                _estimateMarkStateSpaceKernelMethod = (_) => empty(0);

                break;

            case EstimationMethod.KernelCompression:
                _observationEstimation = new KernelCompression(
                    bandwidth: observationBandwidth, 
                    dimensions: _stateSpace.Dimensions, 
                    distanceThreshold: distanceThreshold,
                    kernelLimit: kernelLimit,
                    device: device,
                    scalarType: scalarType
                );

                var bandwidth = observationBandwidth.Concat(markBandwidth).ToArray();
                var jointDimensions = _stateSpace.Dimensions + _markDimensions;

                for (int i = 0; i < _markChannels; i++)
                {
                    _channelEstimation[i] = new KernelCompression(
                        bandwidth: observationBandwidth, 
                        dimensions: _stateSpace.Dimensions, 
                        distanceThreshold: distanceThreshold,
                        kernelLimit: kernelLimit,
                        device: device,
                        scalarType: scalarType
                    );

                    _markEstimation[i] = new KernelCompression(
                        bandwidth: bandwidth, 
                        dimensions: jointDimensions, 
                        distanceThreshold: distanceThreshold,
                        kernelLimit: kernelLimit,
                        device: device,
                        scalarType: scalarType
                    );
                }

                _markFitMethod = FitMarksUnfactoredMethod;
                _estimateMarkConditionalIntensityMethod = EstimateMarksUnfactoredMethod;
                _estimateMarkStateSpaceKernelMethod = (IEstimation estimation) => 
                    estimation.Estimate(_stateSpace.Points, 0, _stateSpace.Dimensions);

                break;

            default:
                throw new ArgumentException("Invalid estimation method.");
        };

        _estimations = [_observationEstimation, .. _channelEstimation, .. _markEstimation];
    }

    private void FitMarksFactoredMethod(
        int i, 
        Tensor observations, 
        Tensor marks
    )
    {
        _markEstimation[i].Fit(marks);
    }

    private void FitMarksUnfactoredMethod(
        int i, 
        Tensor observations, 
        Tensor marks
    )
    {
        _markEstimation[i].Fit(concat([observations, marks], dim: 1));
    }

    private Tensor EstimateMarksFactoredMethod(
        int i,
        Tensor marks
    )
    {
        using var _ = NewDisposeScope();
        var markKernelEstimate = _markEstimation[i].Estimate(marks);
        var markDensity = markKernelEstimate.matmul(_channelEstimates[i].T)
            / markKernelEstimate.shape[1];
        return (_rates[i] + markDensity.log() - _observationDensity)
            .nan_to_num()
            .MoveToOuterDisposeScope();
    }

    private Tensor EstimateMarksUnfactoredMethod(
        int i,
        Tensor marks
    )
    {
        using var _ = NewDisposeScope();
        var markKernelEstimate = _markEstimation[i].Estimate(marks, _stateSpace.Dimensions);
        var markDensity = markKernelEstimate.matmul(_markStateSpaceKernelEstimates[i].T)
            / markKernelEstimate.shape[1];
        return (_rates[i] + markDensity.log() - _observationDensity)
            .nan_to_num()
            .MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
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
            _spikeCounts = (marks.sum(dim: 1) > 0)
                .sum(dim: 0)
                .to(_device);
            _samples = tensor(observations.shape[0], device: _device);

        }
        else
        {
            _spikeCounts += (marks.sum(dim: 1) > 0)
                .sum(dim: 0);
            _samples += observations.shape[0];
        }

        _rates = (_spikeCounts.log() - _samples.log())
            .MoveToOuterDisposeScope();
        
        var mask = ~marks.isnan().all(dim: 1);

        for (int i = 0; i < _markChannels; i++)
        {
            if (mask[TensorIndex.Colon, i].sum().item<long>() == 0)
            {
                continue;
            }

            var channelObservation = observations[mask[TensorIndex.Colon, i]];
            var markObservation = marks[TensorIndex.Tensor(mask[TensorIndex.Colon, i]), TensorIndex.Colon, i];
            _channelEstimation[i].Fit(channelObservation);
            _markFitMethod(i, channelObservation, markObservation);
        }

        _updateConditionalIntensities = true;
        _channelConditionalIntensities = Evaluate()
            .First()
            .MoveToOuterDisposeScope();
    }

    private Tensor EvaluateMarkConditionalIntensities(Tensor inputs)
    {
        using var _ = NewDisposeScope();

         var markConditionalIntensities = zeros(
            [_markChannels, inputs.shape[0], _stateSpace.Points.shape[0]],
            device: _device,
            dtype: _scalarType
        );

        var mask = ~inputs.isnan().all(dim: 1);

        for (int i = 0; i < _markChannels; i++)
        {
            if (mask[TensorIndex.Colon, i].sum().item<long>() == 0)
            {
                continue;
            }

            var marks = inputs[TensorIndex.Tensor(mask[TensorIndex.Colon, i]), TensorIndex.Colon, i];
            markConditionalIntensities[i, TensorIndex.Tensor(mask[TensorIndex.Colon, i])] = _estimateMarkConditionalIntensityMethod(
                i,
                marks
            );
        }

        return markConditionalIntensities
            .MoveToOuterDisposeScope();
    }

    private Tensor EvaluateChannelConditionalIntensities()
    {
        using var _ = NewDisposeScope();

        _observationDensity = _observationEstimation.Evaluate(_stateSpace.Points)
            .log();

        var channelConditionalIntensities = zeros(
            [_markChannels, _stateSpace.Points.shape[0]],
            device: _device,
            dtype: _scalarType
        );

        _channelEstimates = new Tensor[_markChannels];
        _markStateSpaceKernelEstimates = new Tensor[_markChannels];

        for (int i = 0; i < _markChannels; i++)
        {
            _channelEstimates[i] = _channelEstimation[i].Estimate(_stateSpace.Points)
                .MoveToOuterDisposeScope();

            var channelDensity = _channelEstimation[i].Normalize(_channelEstimates[i])
                .log();

            channelConditionalIntensities[i] = exp(_rates[i] + channelDensity - _observationDensity);
            
            _markStateSpaceKernelEstimates[i] = _estimateMarkStateSpaceKernelMethod(_markEstimation[i])
                .MoveToOuterDisposeScope();
        }

        _observationDensity.MoveToOuterDisposeScope();
        _updateConditionalIntensities = false;

        return channelConditionalIntensities
            .MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
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

        _conditionalIntensities = [_channelConditionalIntensities, _markConditionalIntensities];

        return _conditionalIntensities;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _observationEstimation.Dispose();
        foreach (var estimation in _channelEstimation)
        {
            estimation.Dispose();
        }
        foreach (var estimation in _markEstimation)
        {
            estimation.Dispose();
        }
        _estimations = [];
        _updateConditionalIntensities = true;
        _conditionalIntensities = [empty(0)];
        _markConditionalIntensities.Dispose();
        _channelConditionalIntensities.Dispose();
        _spikeCounts.Dispose();
        _samples.Dispose();
        _rates.Dispose();
    }
}
