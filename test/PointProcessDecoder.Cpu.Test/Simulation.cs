using PointProcessDecoder.Test.Common;

namespace PointProcessDecoder.Cpu.Test;

[TestClass]
public class Simulation
{
    [TestMethod]
    public void TestSpikes()
    {
        TestSimulation.SpikingNeurons1D();
        TestSimulation.SpikingNeurons2D();
        TestSimulation.SpikingNeurons2DFirstAndLastSteps();
        Assert.IsTrue(true);
    }

    [TestMethod]
    public void TestMarks()
    {
        TestSimulation.Marks1D(
            noiseScale: 1.9
        );
        Assert.IsTrue(true);
    }
}