using static TorchSharp.torch;

namespace PointProcessDecoder.Core;

public interface IDecoder : IDisposable
{
    public Device Device { get; }
    public ScalarType ScalarType { get; }
    public Decoder.DecoderType DecoderType { get; }
    public Tensor InitialState { get; }
    public IStateTransitions Transitions { get; }
    public Tensor Decode(Tensor input, Tensor likelihood);
}
