using PointProcessDecoder.Test.Common;

namespace PointProcessDecoder.Cpu.Test;

[TestClass]
public class TestSimulation
{
    [TestMethod]
    public void TestSpikes()
    {
        SimulationUtilities.SpikingNeurons1D();
        SimulationUtilities.SpikingNeurons2D();
        SimulationUtilities.SpikingNeurons2DFirstAndLastSteps();
    }

    [TestMethod]
    public void TestMarks()
    {
        SimulationUtilities.Marks1D();
    }
}