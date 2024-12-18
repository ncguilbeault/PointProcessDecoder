using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Likelihood;

public static class PoissonLikelihood
{
    public static Tensor LogLikelihood(Tensor inputs, Tensor conditionalIntensities)
    {
        using var _ = NewDisposeScope();
        var logLikelihood = xlogy(inputs.unsqueeze(1), conditionalIntensities.unsqueeze(0)) - conditionalIntensities.unsqueeze(0);
        return logLikelihood.sum(dim: -1).MoveToOuterDisposeScope();
    }
}
