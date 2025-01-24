using static TorchSharp.torch;

namespace PointProcessDecoder.Core.Likelihood;

public class PoissonLikelihood : ILikelihood
{
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
}
