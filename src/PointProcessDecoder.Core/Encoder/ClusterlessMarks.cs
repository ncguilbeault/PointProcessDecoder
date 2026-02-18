using PointProcessDecoder.Core.Estimation;
using TorchSharp;
using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Encoder;

/// <summary>
/// Represents a clusterless mark encoder.
/// </summary>
public class ClusterlessMarks : ModelComponent, IEncoder
{
    private readonly Device _device;
    /// <inheritdoc/>
    public override Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <inheritdoc/>
    public override ScalarType ScalarType => _scalarType;

    /// <inheritdoc/>
    public EncoderType EncoderType => EncoderType.ClusterlessMarks;

    /// <inheritdoc/>
    public Tensor[] Intensities => [_channelIntensities, _markIntensities];

    /// <inheritdoc/>
    public IEstimation[] Estimations => [_covariateEstimation, .. _markEstimation];

    private readonly IEstimation _covariateEstimation;
    private readonly IEstimation[] _markEstimation;

    private readonly IStateSpace _stateSpace;
    private bool _updateIntensities = true;
    private Tensor _markIntensities = empty(0);
    private Tensor _channelIntensities = empty(0);
    private Tensor _covariateDensity = empty(0);
    private readonly Tensor[] _channelEstimates = [];

    private Tensor _spikeCounts = empty(0);
    private Tensor _samples = empty(0);
    private Tensor _rates = empty(0);

    private readonly int _markDimensions;
    private readonly int _numChannels;

    /// <summary>
    /// Initializes a new instance of the <see cref="ClusterlessMarks"/> class.
    /// </summary>
    /// <param name="estimationMethod"></param>
    /// <param name="covariateBandwidth"></param>
    /// <param name="markDimensions"></param>
    /// <param name="numChannels"></param>
    /// <param name="markBandwidth"></param>
    /// <param name="stateSpace"></param>
    /// <param name="distanceThreshold"></param>
    /// <param name="device"></param>
    /// <param name="scalarType"></param>
    /// <exception cref="ArgumentException"></exception>
    public ClusterlessMarks(
        EstimationMethod estimationMethod,
        double[] covariateBandwidth,
        int markDimensions,
        int numChannels,
        double[] markBandwidth,
        IStateSpace stateSpace,
        double? distanceThreshold = null,
        int? kernelLimit = null,
        Device? device = null,
        ScalarType? scalarType = null
    )
    {
        if (numChannels < 1)
        {
            throw new ArgumentException("The number of mark channels must be greater than 0.", nameof(numChannels));
        }

        if (markDimensions < 1)
        {
            throw new ArgumentException("The number of mark dimensions must be greater than 0.", nameof(markDimensions));
        }

        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _markDimensions = markDimensions;
        _numChannels = numChannels;
        _stateSpace = stateSpace;

        _markEstimation = new IEstimation[_numChannels];
        _channelEstimates = new Tensor[_numChannels];

        var bandwidth = covariateBandwidth.Concat(markBandwidth).ToArray();
        var jointDimensions = _stateSpace.Dimensions + _markDimensions;

        switch (estimationMethod)
        {
            case EstimationMethod.KernelDensity:

                _covariateEstimation = new KernelDensity(
                    bandwidth: covariateBandwidth,
                    dimensions: _stateSpace.Dimensions,
                    kernelLimit: kernelLimit,
                    device: device,
                    scalarType: scalarType
                );

                for (int i = 0; i < _numChannels; i++)
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

                _covariateEstimation = new KernelCompression(
                    bandwidth: covariateBandwidth,
                    dimensions: _stateSpace.Dimensions,
                    distanceThreshold: distanceThreshold,
                    kernelLimit: kernelLimit,
                    device: device,
                    scalarType: scalarType
                );

                for (int i = 0; i < _numChannels; i++)
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
                throw new ArgumentException("Invalid estimation method.", nameof(estimationMethod));
        };
    }

    /// <inheritdoc/>
    public void Encode(Tensor covariates, Tensor observations)
    {
        if (observations.ndim != 3)
        {
            throw new ArgumentException("The marks tensor must have 3 dimensions (numSamples, markDimensions, numChannels).", nameof(observations));
        }

        if (covariates.ndim != 2)
        {
            throw new ArgumentException("The covariates tensor must have 2 dimensions (numSamples, covariateDimensions).", nameof(observations));
        }

        var marksShape = observations.shape;
        var numMarkSamples = marksShape[0];
        var markDimensions = marksShape[1];
        var numChannels = marksShape[2];

        var covariatesShape = covariates.shape;
        var numCovariateSamples = covariatesShape[0];
        var covariateDimensions = covariatesShape[1];

        if (markDimensions != _markDimensions)
        {
            throw new ArgumentException("The number of mark dimensions must match the shape of the marks tensor on dimension 1.", nameof(observations));
        }

        if (numChannels != _numChannels)
        {
            throw new ArgumentException("The number of mark channels must match the shape of the marks tensor on dimension 2.", nameof(observations));
        }

        if (covariateDimensions != _stateSpace.Dimensions)
        {
            throw new ArgumentException("The number of covariate dimensions must match the dimensions of the state space.", nameof(covariates));
        }

        if (numCovariateSamples != numMarkSamples && numCovariateSamples != 1)
        {
            throw new ArgumentException("The number of samples in the covariates and marks tensors must match, unless covariates has only one sample.", nameof(covariates));
        }

        _covariateEstimation.Fit(covariates);

        if (_spikeCounts.numel() == 0)
        {
            _spikeCounts = (~observations.isnan())
                .any(dim: 1)
                .sum(dim: 0)
                .to(_device);
            _samples = numMarkSamples;
        }
        else
        {
            _spikeCounts += (~observations.isnan())
                .any(dim: 1)
                .sum(dim: 0);
            _samples += numMarkSamples;
        }

        _rates = _spikeCounts.log() - _samples.log();

        var mask = ~observations.isnan().all(dim: 1);

        for (int i = 0; i < _numChannels; i++)
        {
            if ((~mask[TensorIndex.Colon, i].any()).item<bool>())
            {
                continue;
            }

            if (numCovariateSamples == 1)
            {
                covariates = covariates.expand(numMarkSamples, -1);
            }

            _markEstimation[i].Fit(
                concat([
                    covariates[mask[TensorIndex.Colon, i]],
                    observations[TensorIndex.Tensor(mask[TensorIndex.Colon, i]), TensorIndex.Colon, i]
                ], dim: 1)
            );
        }

        _updateIntensities = true;
        Evaluate();
    }

    private void EvaluateMarkIntensities(Tensor observations)
    {
        using var _ = NewDisposeScope();

        var numSamples = observations.size(0);

        _markIntensities = zeros(
            [_numChannels, numSamples, _stateSpace.Points.size(0)],
            device: _device,
            dtype: _scalarType
        );

        var mask = ~observations.isnan().all(dim: 1);

        for (int i = 0; i < _numChannels; i++)
        {
            if ((~mask[TensorIndex.Colon, i].any()).item<bool>())
            {
                continue;
            }

            var markKernelEstimate = _markEstimation[i].Estimate(
                observations[TensorIndex.Tensor(mask[TensorIndex.Colon, i]), TensorIndex.Colon, i],
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

            _markIntensities[i, TensorIndex.Tensor(mask[TensorIndex.Colon, i])] = _rates[i] + markDensity - _covariateDensity;
        }

        _markIntensities.MoveToOuterDisposeScope();
    }

    private void EvaluateChannelIntensities()
    {
        using var _ = NewDisposeScope();

        _covariateDensity = _covariateEstimation.Evaluate(_stateSpace.Points)
            .log()
            .nan_to_num()
            .MoveToOuterDisposeScope();

        _channelIntensities = zeros(
            [_numChannels, _stateSpace.Points.size(0)],
            device: _device,
            dtype: _scalarType
        );

        for (int i = 0; i < _numChannels; i++)
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

            _channelIntensities[i] = _rates[i] + channelDensity - _covariateDensity;
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
        _covariateDensity.Save(Path.Combine(path, "covariateDensity.bin"));
        _channelIntensities.Save(Path.Combine(path, "channelIntensities.bin"));

        var covariateEstimationPath = Path.Combine(path, $"covariateEstimation");

        if (!Directory.Exists(covariateEstimationPath))
        {
            Directory.CreateDirectory(covariateEstimationPath);
        }

        _covariateEstimation.Save(covariateEstimationPath);

        for (int i = 0; i < _numChannels; i++)
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
        _covariateDensity = Tensor.Load(Path.Combine(path, "covariateDensity.bin")).to(_device);
        _channelIntensities = Tensor.Load(Path.Combine(path, "channelIntensities.bin")).to(_device);

        var covariateEstimationPath = Path.Combine(path, $"covariateEstimation");

        if (!Directory.Exists(covariateEstimationPath))
        {
            throw new ArgumentException("The covariate estimation directory does not exist within the specified base path.", nameof(basePath));
        }

        _covariateEstimation.Load(covariateEstimationPath);

        for (int i = 0; i < _numChannels; i++)
        {
            var channelEstimationPath = Path.Combine(path, $"channelEstimation{i}");

            if (!Directory.Exists(channelEstimationPath))
            {
                throw new ArgumentException("The channel estimation directory does not exist within the specified base path.", nameof(basePath));
            }

            var markEstimationPath = Path.Combine(path, $"markEstimation{i}");

            if (!Directory.Exists(markEstimationPath))
            {
                throw new ArgumentException("The mark estimation directory does not exist within the specified base path.", nameof(basePath));
            }

            _markEstimation[i].Load(markEstimationPath);
            _channelEstimates[i] = Tensor.Load(Path.Combine(path, $"channelEstimates{i}.bin")).to(_device);
        }

        return this;
    }
}
