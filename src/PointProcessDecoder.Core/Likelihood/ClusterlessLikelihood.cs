using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Likelihood;

public class ClusterlessLikelihood(bool ignoreNoSpikes = false) : ILikelihood
{
    private bool _ignoreNoSpikes = ignoreNoSpikes;
    private Tensor _noSpikeLikelihood = ignoreNoSpikes ? zeros(1) : ones(1);
    public bool IgnoreNoSpikes
    {
        get => _ignoreNoSpikes;
        set 
        {
            _ignoreNoSpikes = value;
            _noSpikeLikelihood = _ignoreNoSpikes ? zeros(1) : ones(1);
        }
    }

    public LikelihoodType LikelihoodType => LikelihoodType.Clusterless;

    public Tensor LogLikelihood(
        Tensor inputs, 
        IEnumerable<Tensor> conditionalIntensities
    )
    {
        using var _ = NewDisposeScope();
        var channelConditionalIntensities = conditionalIntensities.ElementAt(0);
        var markConditionalIntensities = conditionalIntensities.ElementAt(1);
        var logLikelihood = markConditionalIntensities.sum(dim: 0) - channelConditionalIntensities.sum(dim: 0) * _noSpikeLikelihood;
        logLikelihood -= logLikelihood.max(dim: -1, keepdim: true)
            .values;
        return logLikelihood
            .exp()
            .nan_to_num()
            .MoveToOuterDisposeScope();
    }
}
