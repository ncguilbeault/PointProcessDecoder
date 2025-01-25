using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Likelihood;

public class ClusterlessLikelihood(bool ignoreNoSpikes = false) : ILikelihood
{
    private readonly bool _ignoreNoSpikes = ignoreNoSpikes;

    public Tensor LogLikelihood(
        Tensor inputs, 
        IEnumerable<Tensor> conditionalIntensities
    )
    {
        using var _ = NewDisposeScope();
        var channelConditionalIntensities = conditionalIntensities.ElementAt(0);
        var markConditionalIntensities = conditionalIntensities.ElementAt(1);
        var logLikelihood = markConditionalIntensities.sum(dim: 0) - channelConditionalIntensities.sum(dim: 0) * _ignoreNoSpikes;
        logLikelihood -= logLikelihood.max(dim: -1, keepdim: true)
            .values;
        return logLikelihood
            .exp()
            .nan_to_num()
            .MoveToOuterDisposeScope();
    }
}
