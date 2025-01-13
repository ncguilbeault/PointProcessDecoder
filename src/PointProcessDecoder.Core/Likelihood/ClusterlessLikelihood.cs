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
        var mask = ~inputs.isnan()
            .all(dim: 1)
            .unsqueeze(0).T;
        var logLikelihood = (markConditionalIntensities * mask - channelConditionalIntensities.unsqueeze(1))
            .nan_to_num()
            .sum(dim: 0);
        logLikelihood -= logLikelihood.max(dim: 0, keepdim: true)
            .values;
        return logLikelihood.MoveToOuterDisposeScope();
    }
}
