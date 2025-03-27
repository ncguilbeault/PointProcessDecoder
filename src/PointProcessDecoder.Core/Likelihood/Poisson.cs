using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Likelihood;

/// <summary>
/// Represents the Poisson likelihood of the model.
/// Expected to be used when the encoder is set to the <see cref="Encoder.EncoderType.SortedSpikeEncoder"/>.
/// </summary>
/// <param name="device"></param>
/// <param name="scalarType"></param>
public class Poisson(
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
    /// Whether to ignore the contribution of no spikes to the likelihood.
    /// </summary>
    public bool IgnoreNoSpikes
    {
        get => _ignoreNoSpikes;
        set => _ignoreNoSpikes = value;
    }

    /// <inheritdoc />
    public LikelihoodType LikelihoodType => LikelihoodType.Poisson;

    /// <inheritdoc />
    public Tensor Likelihood(
        Tensor inputs, 
        IEnumerable<Tensor> intensities
    )
    {
        using var _ = NewDisposeScope();

        var intensity = intensities.First()
            .unsqueeze(0);

        var likelihood = inputs.unsqueeze(-1) 
            * intensity;
        
        if (!_ignoreNoSpikes) {
            likelihood -= intensity
                .exp();
        }

        likelihood = likelihood
            .sum(dim: 1);

        likelihood -= likelihood.max(dim: -1, keepdim: true).values;

        likelihood = likelihood
            .exp();

        likelihood /= likelihood
            .sum(dim: -1, keepdim: true);

        return likelihood
            .MoveToOuterDisposeScope();
    }
}
