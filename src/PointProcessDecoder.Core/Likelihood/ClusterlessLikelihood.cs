using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Likelihood;

public class ClusterlessLikelihood : ILikelihood
{
    public Tensor LogLikelihood(
        Tensor inputs, 
        IEnumerable<Tensor> conditionalIntensities
    )
    {
        using var _ = NewDisposeScope();
        var channelConditionalIntensities = conditionalIntensities.ElementAt(0);
        var markConditionalIntensities = conditionalIntensities.ElementAt(1);
        var logLikelihood = (markConditionalIntensities - channelConditionalIntensities.unsqueeze(1))
            .nan_to_num()
            .sum(dim: 0);
        logLikelihood -= logLikelihood.max(dim: -1, keepdim: true)
            .values;
        return logLikelihood.MoveToOuterDisposeScope();
    }
}
