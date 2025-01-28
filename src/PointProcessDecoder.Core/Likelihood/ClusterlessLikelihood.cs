using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Likelihood;

public class ClusterlessLikelihood : ILikelihood
{
    private readonly Device _device;
    /// <inheritdoc/>
    public Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <inheritdoc/>
    public ScalarType ScalarType => _scalarType;

    private bool _ignoreNoSpikes;
    private Tensor _noSpikeLikelihood;
    /// <summary>
    /// Whether to ignore the contribution of channels with no spikes to the likelihood.
    /// </summary>
    public bool IgnoreNoSpikes
    {
        get => _ignoreNoSpikes;
        set 
        {
            _ignoreNoSpikes = value;
            _noSpikeLikelihood = _ignoreNoSpikes ? 
                zeros(1, device: _device, dtype: _scalarType) 
                : ones(1, dtype: _scalarType, device: _device);
        }
    }

    /// <inheritdoc />
    public LikelihoodType LikelihoodType => LikelihoodType.Clusterless;

    /// <summary>
    /// Initializes a new instance of the <see cref="ClusterlessLikelihood"/> class.
    /// </summary>
    /// <param name="device"></param>
    /// <param name="scalarType"></param>
    /// <param name="ignoreNoSpikes"></param>
    public ClusterlessLikelihood(
        Device? device = null,
        ScalarType? scalarType = null,
        bool ignoreNoSpikes = false
    )
    {
        _device = device ?? CPU;
        _scalarType = scalarType ?? ScalarType.Float32;
        _ignoreNoSpikes = ignoreNoSpikes;
        _noSpikeLikelihood = _ignoreNoSpikes ? 
            zeros(1, dtype: _scalarType, device: _device) 
            : ones(1, dtype: _scalarType, device: _device);
    }

    /// <inheritdoc />
    public Tensor LogLikelihood(
        Tensor inputs, 
        IEnumerable<Tensor> conditionalIntensities
    )
    {
        using var _ = NewDisposeScope();
        var channelConditionalIntensities = conditionalIntensities.ElementAt(0);
        var markConditionalIntensities = conditionalIntensities.ElementAt(1);
        var logLikelihood = (markConditionalIntensities.sum(dim: 0) - channelConditionalIntensities.sum(dim: 0) * _noSpikeLikelihood)
            .nan_to_num();
        logLikelihood -= logLikelihood.max(dim: -1, keepdim: true)
            .values;
        return logLikelihood
            .exp()
            .nan_to_num()
            .MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _noSpikeLikelihood.Dispose();
    }
}
