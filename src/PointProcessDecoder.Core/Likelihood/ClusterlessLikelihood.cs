using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Likelihood;

/// <summary>
/// Represents a clusterless likelihood.
/// Expected to be used when the encoder is set to the <see cref="Encoder.EncoderType.ClusterlessMarkEncoder"/>.
/// </summary>
public class ClusterlessLikelihood(
    Device? device = null,
    ScalarType? scalarType = null,
    bool ignoreNoSpikes = false
) : ModelComponent, ILikelihood
{
    private readonly Device _device = device ?? CPU;
    /// <inheritdoc/>
    public override Device Device => _device;

    private readonly ScalarType _scalarType = scalarType ?? ScalarType.Float32;
    /// <inheritdoc/>
    public override ScalarType ScalarType => _scalarType;

    private bool _ignoreNoSpikes = ignoreNoSpikes;
    /// <summary>
    /// Whether to ignore the contribution of channels with no spikes to the likelihood.
    /// </summary>
    public bool IgnoreNoSpikes
    {
        get => _ignoreNoSpikes;
        set => _ignoreNoSpikes = value;
    }

    /// <inheritdoc />
    public LikelihoodType LikelihoodType => LikelihoodType.Clusterless;

    /// <inheritdoc />
    public Tensor Likelihood(
        Tensor inputs, 
        IEnumerable<Tensor> intensities
    )
    {
        using var _ = NewDisposeScope();

        var channelIntensities = intensities.ElementAt(0);
        var markIntensities = intensities.ElementAt(1);

        var likelihood = markIntensities;

        if (!_ignoreNoSpikes)
        {
            likelihood -= channelIntensities
                .unsqueeze(1)
                .exp();
        }
        
        likelihood = likelihood
            .sum(dim: 0);

        likelihood -= likelihood.max(dim: -1, keepdim: true).values;

        likelihood = likelihood.exp();

        likelihood /= likelihood
            .sum(dim: -1, keepdim: true);

        return likelihood
            .MoveToOuterDisposeScope();
    }
}
