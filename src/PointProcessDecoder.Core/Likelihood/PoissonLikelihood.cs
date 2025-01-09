using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Likelihood;

public static class PoissonLikelihood
{
    public static Tensor LogLikelihood(
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
        return logLikelihood.MoveToOuterDisposeScope();
    }
}
