using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Likelihood;

/// <summary>
/// Represents the Poisson likelihood of the model.
/// Expected to be used when the encoder is set to the <see cref="Encoder.EncoderType.SortedSpikeEncoder"/>.
/// </summary>
/// <param name="device"></param>
/// <param name="scalarType"></param>
public class PoissonLikelihood(
    Device? device = null,
    ScalarType? scalarType = null
) : ModelComponent, ILikelihood
{
    private readonly Device _device = device ?? CPU;
    /// <inheritdoc/>
    public override Device Device => _device;

    private readonly ScalarType _scalarType = scalarType ?? ScalarType.Float32;
    /// <inheritdoc/>
    public override ScalarType ScalarType => _scalarType;

    /// <inheritdoc />
    public LikelihoodType LikelihoodType => LikelihoodType.Poisson;

    /// <inheritdoc />
    public Tensor Likelihood(
        Tensor inputs, 
        IEnumerable<Tensor> conditionalIntensities
    )
    {
        using var _ = NewDisposeScope();
        var conditionalIntensity = conditionalIntensities.First();
        var logLikelihood = (inputs
            .unsqueeze(1) 
            * conditionalIntensity 
            - conditionalIntensity.exp())
                .nan_to_num()
                .sum(dim: -1);
        logLikelihood -= logLikelihood
            .max(dim: -1, keepdim: true)
            .values;
        logLikelihood = logLikelihood
            .exp()
            .nan_to_num();
        logLikelihood /= logLikelihood
            .sum(dim: -1, keepdim: true);
        return logLikelihood
            .MoveToOuterDisposeScope();
    }
}
