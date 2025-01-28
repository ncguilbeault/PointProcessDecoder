using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Likelihood;

public class PoissonLikelihood(
    Device? device = null,
    ScalarType? scalarType = null
) : ILikelihood
{
    private readonly Device _device = device ?? CPU;
    /// <inheritdoc/>
    public Device Device => _device;

    private readonly ScalarType _scalarType = scalarType ?? ScalarType.Float32;
    /// <inheritdoc/>
    public ScalarType ScalarType => _scalarType;

    /// <inheritdoc />
    public LikelihoodType LikelihoodType => LikelihoodType.Poisson;

    /// <inheritdoc />
    public Tensor LogLikelihood(
        Tensor inputs, 
        IEnumerable<Tensor> conditionalIntensities
    )
    {
        using var _ = NewDisposeScope();
        var conditionalIntensity = conditionalIntensities.First();
        var conditionalIntensityTensor = conditionalIntensity.flatten(1).T.unsqueeze(0);
        var logLikelihood = (xlogy(inputs.unsqueeze(1), conditionalIntensityTensor) - conditionalIntensityTensor)
            .nan_to_num()
            .sum(dim: -1);
        logLikelihood -= logLikelihood.max(dim: -1, keepdim: true).values;
        return logLikelihood
            .exp()
            .nan_to_num()
            .MoveToOuterDisposeScope();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
    }
}
