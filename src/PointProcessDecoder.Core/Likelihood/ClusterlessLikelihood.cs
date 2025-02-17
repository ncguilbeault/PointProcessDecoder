using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Likelihood;

/// <summary>
/// Represents a clusterless likelihood.
/// Expected to be used when the encoder is set to the <see cref="Encoder.EncoderType.ClusterlessMarkEncoder"/>.
/// </summary>
public class ClusterlessLikelihood : ModelComponent, ILikelihood
{
    private readonly Device _device;
    /// <inheritdoc/>
    public override Device Device => _device;

    private readonly ScalarType _scalarType;
    /// <inheritdoc/>
    public override ScalarType ScalarType => _scalarType;

    private bool _ignoreNoSpikes;
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
    }

    /// <inheritdoc />
    public Tensor Likelihood(
        Tensor inputs, 
        IEnumerable<Tensor> intensities
    )
    {
        using var _ = NewDisposeScope();

        var channelIntensities = intensities.ElementAt(0);
        var markIntensities = intensities.ElementAt(1);

        var likelihood = markIntensities
            .sum(dim: 0);

        if (!_ignoreNoSpikes)
        {
            likelihood -= channelIntensities
                .exp()
                .sum(dim: 0);
        }

        likelihood = likelihood
            .exp();

        likelihood /= likelihood
            .sum(dim: -1, keepdim: true);

        return likelihood
            .MoveToOuterDisposeScope();
    }
}
