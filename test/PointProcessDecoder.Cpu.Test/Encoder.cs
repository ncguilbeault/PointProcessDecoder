using PointProcessDecoder.Test.Common;

namespace PointProcessDecoder.Cpu.Test;

[TestClass]
public class Encoder
{
    [TestMethod]
    public void Test()
    {
        TestEncoder.KernelDensity(
            bandwidth: [5.0],
            numDimensions: 1,
            evaluationSteps: [50],
            steps: 200,
            cycles: 10,
            min: [0],
            max: [100],
            numNeurons: 40,
            placeFieldRadius: 8.0,
            firingThreshold: 0.2
        );

        TestEncoder.KernelDensity(
            bandwidth: [5.0, 5.0],
            numDimensions: 2,
            evaluationSteps: [50, 50],
            steps: 200,
            cycles: 10,
            min: [0, 0],
            max: [100, 100],
            numNeurons: 40,
            placeFieldRadius: 8.0,
            firingThreshold: 0.2,
            modelDirectory: "KernelDensity2D"
        );

        TestEncoder.KernelCompression(
            bandwidth: [5.0],
            numDimensions: 1,
            evaluationSteps: [50],
            steps: 200,
            cycles: 10,
            min: [0],
            max: [100],
            numNeurons: 40,
            placeFieldRadius: 8.0,
            firingThreshold: 0.2,
            distanceThreshold: 1.5
        );

        TestEncoder.KernelCompression(
            bandwidth: [5.0, 5.0],
            numDimensions: 2,
            evaluationSteps: [50, 50],
            steps: 200,
            cycles: 10,
            min: [0, 0],
            max: [100, 100],
            numNeurons: 40,
            placeFieldRadius: 8.0,
            firingThreshold: 0.2,
            distanceThreshold: 1.5,
            modelDirectory: "KernelCompression2D"
        );

        Assert.IsTrue(true);
    }
}