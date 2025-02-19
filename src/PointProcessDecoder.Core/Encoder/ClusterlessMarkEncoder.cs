using PointProcessDecoder.Core.Estimation;
using TorchSharp;
using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Encoder;

/// <summary>
/// Represents a clusterless mark encoder.
/// </summary>
public class ClusterlessMarkEncoder : ModelComponent, IEncoder
{
    private readonly Device _device;
    /// <inheritdoc/>
    public override Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <inheritdoc/>
    public override ScalarType ScalarType => _scalarType;

    /// <inheritdoc/>
    public EncoderType EncoderType => EncoderType.ClusterlessMarkEncoder;

    /// <inheritdoc/>
    public Tensor[] Intensities => [_channelIntensities, _markIntensities];

    /// <inheritdoc/>
    public IEstimation[] Estimations => [_observationEstimation, .. _markEstimation];

    private readonly IEstimation _observationEstimation;
    private readonly IEstimation[] _markEstimation;

    private readonly IStateSpace _stateSpace;
    private bool _updateIntensities = true;
    private Tensor _markIntensities = empty(0);
    private Tensor _channelIntensities = empty(0);
    private Tensor _observationDensity = empty(0);
    private Tensor[] _channelEstimates = [];

    private Tensor _spikeCounts = empty(0);
    private Tensor _samples = empty(0);
    private Tensor _rates = empty(0);

    private readonly int _markDimensions;
    private readonly int _markChannels;

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

        _markEstimation = new IEstimation[_markChannels];
        _channelEstimates = new Tensor[_markChannels];

        var bandwidth = observationBandwidth.Concat(markBandwidth).ToArray();
        var jointDimensions = _stateSpace.Dimensions + _markDimensions;

        switch (estimationMethod)
        {
            case EstimationMethod.KernelDensity:

                _observationEstimation = new KernelDensity(
                    bandwidth: observationBandwidth, 
                    dimensions: _stateSpace.Dimensions, 
                    kernelLimit: kernelLimit,
                    device: device,
                    scalarType: scalarType
                );

                for (int i = 0; i < _markChannels; i++)
                {
                    _markEstimation[i] = new KernelDensity(
                        bandwidth: bandwidth, 
                        dimensions: jointDimensions,
                        kernelLimit: kernelLimit,
                        device: device,
                        scalarType: scalarType
                    );
                }

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

                for (int i = 0; i < _markChannels; i++)
                {
                    _markEstimation[i] = new KernelCompression(
                        bandwidth: bandwidth, 
                        dimensions: jointDimensions, 
                        distanceThreshold: distanceThreshold,
                        kernelLimit: kernelLimit,
                        device: device,
                        scalarType: scalarType
                    );
                }

                break;

            default:
                throw new ArgumentException("Invalid estimation method.");
        };
    }

    /// <inheritdoc/>
    public void Encode(Tensor observations, Tensor marks)
    {
        if (marks.size(1) != _markDimensions)
        {
            throw new ArgumentException("The number of mark dimensions must match the shape of the marks tensor on dimension 1.");
        }

        if (marks.size(2) != _markChannels)
        {
            throw new ArgumentException("The number of mark channels must match the shape of the marks tensor on dimension 2.");
        }

        if (observations.size(1) != _stateSpace.Dimensions)
        {
            throw new ArgumentException("The number of observation dimensions must match the dimensions of the state space.");
        }

        _observationEstimation.Fit(observations);

        if (_spikeCounts.numel() == 0)
        {
            _spikeCounts = (~marks.isnan())
                .any(dim: 1)
                .sum(dim: 0)
                .to(_device);
            _samples = tensor(observations.size(0), device: _device);

        }
        else
        {
            _spikeCounts += (~marks.isnan())
                .any(dim: 1)
                .sum(dim: 0);
            _samples += observations.size(0);
        }

        _rates = _spikeCounts.log() - _samples.log();
        
        var mask = ~marks.isnan().all(dim: 1);

        for (int i = 0; i < _markChannels; i++)
        {
            if ((~mask[TensorIndex.Colon, i].any()).item<bool>())
            {
                continue;
            }

            _markEstimation[i].Fit(
                concat([
                    observations[mask[TensorIndex.Colon, i]], 
                    marks[TensorIndex.Tensor(mask[TensorIndex.Colon, i]), TensorIndex.Colon, i]
                ], dim: 1)
            );
        }

        _updateIntensities = true;
        Evaluate();
    }

    private void EvaluateMarkIntensities(Tensor inputs)
    {
        using var _ = NewDisposeScope();

        _markIntensities = zeros(
            [_markChannels, inputs.size(0), _stateSpace.Points.size(0)],
            device: _device,
            dtype: _scalarType
        );

        var mask = ~inputs.isnan().all(dim: 1);

        for (int i = 0; i < _markChannels; i++)
        {
            if ((~mask[TensorIndex.Colon, i].any()).item<bool>())
            {
                continue;
            }

            var markKernelEstimate = _markEstimation[i].Estimate(
                inputs[TensorIndex.Tensor(mask[TensorIndex.Colon, i]), TensorIndex.Colon, i],
                _stateSpace.Dimensions
            );

            if (markKernelEstimate.numel() == 0)
            {
                continue;
            }

            var markDensity = markKernelEstimate.matmul(_channelEstimates[i].T);
            markDensity /= markDensity.sum(dim: 1, keepdim: true);
            markDensity = markDensity
                .log()
                .nan_to_num();

            _markIntensities[i, TensorIndex.Tensor(mask[TensorIndex.Colon, i])] = _rates[i] + markDensity - _observationDensity;
        }

        _markIntensities.MoveToOuterDisposeScope();
    }

    private void EvaluateChannelIntensities()
    {
        using var _ = NewDisposeScope();

        _observationDensity = _observationEstimation.Evaluate(_stateSpace.Points)
            .log()
            .nan_to_num()
            .MoveToOuterDisposeScope();

        _channelIntensities = zeros(
            [_markChannels, _stateSpace.Points.size(0)],
            device: _device,
            dtype: _scalarType
        );

        for (int i = 0; i < _markChannels; i++)
        {
            _channelEstimates[i] = _markEstimation[i].Estimate(_stateSpace.Points, 0, _stateSpace.Dimensions)
                .MoveToOuterDisposeScope();

            if (_channelEstimates[i].numel() == 0)
            {
                continue;
            }

            var channelDensity = _markEstimation[i].Normalize(_channelEstimates[i])
                .log()
                .nan_to_num();

            _channelIntensities[i] = _rates[i] + channelDensity - _observationDensity;
        }

        _channelIntensities.MoveToOuterDisposeScope();
        _updateIntensities = false;
    }

    /// <inheritdoc/>
    public IEnumerable<Tensor> Evaluate(params Tensor[] inputs)
    {
        if (_updateIntensities)
        {
            EvaluateChannelIntensities();
        }

        if (inputs.Length > 0)
        {
            EvaluateMarkIntensities(inputs[0]);
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
        _observationDensity.Save(Path.Combine(path, "observationDensity.bin"));
        _channelIntensities.Save(Path.Combine(path, "channelIntensities.bin"));

        var observationEstimationPath = Path.Combine(path, $"observationEstimation");

        if (!Directory.Exists(observationEstimationPath))
        {
            Directory.CreateDirectory(observationEstimationPath);
        }

        _observationEstimation.Save(observationEstimationPath);

        for (int i = 0; i < _markChannels; i++)
        {
            var channelEstimationPath = Path.Combine(path, $"channelEstimation{i}");

            if (!Directory.Exists(channelEstimationPath))
            {
                Directory.CreateDirectory(channelEstimationPath);
            }

            var markEstimationPath = Path.Combine(path, $"markEstimation{i}");

            if (!Directory.Exists(markEstimationPath))
            {
                Directory.CreateDirectory(markEstimationPath);
            }

            _markEstimation[i].Save(markEstimationPath);
            _channelEstimates[i].Save(Path.Combine(path, $"channelEstimates{i}.bin"));
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
        _observationDensity = Tensor.Load(Path.Combine(path, "observationDensity.bin")).to(_device);
        _channelIntensities = Tensor.Load(Path.Combine(path, "channelIntensities.bin")).to(_device);

        var observationEstimationPath = Path.Combine(path, $"observationEstimation");

        if (!Directory.Exists(observationEstimationPath))
        {
            throw new ArgumentException("The observation estimation directory does not exist.");
        }

        _observationEstimation.Load(observationEstimationPath);

        for (int i = 0; i < _markChannels; i++)
        {
            var channelEstimationPath = Path.Combine(path, $"channelEstimation{i}");

            if (!Directory.Exists(channelEstimationPath))
            {
                throw new ArgumentException("The channel estimation directory does not exist.");
            }

            var markEstimationPath = Path.Combine(path, $"markEstimation{i}");

            if (!Directory.Exists(markEstimationPath))
            {
                throw new ArgumentException("The mark estimation directory does not exist.");
            }

            _markEstimation[i].Load(markEstimationPath);

            _channelEstimates[i] = Tensor.Load(Path.Combine(path, $"channelEstimates{i}.bin")).to(_device);
        }

        return this;
    }
}
