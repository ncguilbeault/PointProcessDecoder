using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Likelihood;

public static class ClusterlessLikelihood
{
    public static Tensor LogLikelihood(Tensor inputs, IEnumerable<Tensor> conditionalIntensities)
    {
        using var _ = NewDisposeScope();
        var channelConditionalIntensities = conditionalIntensities.ElementAt(0);
        var markConditionalIntensities = conditionalIntensities.ElementAt(1);
        var logLikelihood = (markConditionalIntensities - exp(channelConditionalIntensities).unsqueeze(1))
            .nan_to_num()
            .sum(dim: 0);
        logLikelihood -= logLikelihood.max(dim: 0).values;
        return logLikelihood.MoveToOuterDisposeScope();
    }
}
