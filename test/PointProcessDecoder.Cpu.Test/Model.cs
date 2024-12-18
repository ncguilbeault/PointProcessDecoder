using PointProcessDecoder.Core.Transitions;
using PointProcessDecoder.Core.Estimation;
using PointProcessDecoder.Test.Common;

namespace PointProcessDecoder.Cpu.Test;

[TestClass]
public class Model
{
    [TestMethod]
    public void TestPointProcessModelUniformDensity()
    {
        TestModel.BayesianStateSpaceSortedUnitsSimulatedData();
    }

    [TestMethod]
    public void TestPointProcessModelRandomWalkDensity()
    {
        TestModel.BayesianStateSpaceSortedUnitsSimulatedData(
            transitionsType: TransitionsType.RandomWalk,
            sigma: [0.1],
            modelDirectory: "SimulatedData1D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelRandomWalkCompression()
    {
        TestModel.BayesianStateSpaceSortedUnitsSimulatedData(
            transitionsType: TransitionsType.RandomWalk,
            estimationMethod: EstimationMethod.KernelCompression,
            sigma: [0.1],
            distanceThreshold: 1.5,
            modelDirectory: "SimulatedData1D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelUniformDensity2D()
    {
        TestModel.BayesianStateSpaceSortedUnitsSimulatedData(
            bandwidth: [5.0, 5.0],
            dimensions: 2,
            evaluationSteps: [50, 50],
            min: [0, 0],
            max: [100, 100],
            scale: 0.1,
            modelDirectory: "SimulatedData2D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelRandomWalkDensity2D()
    {
        TestModel.BayesianStateSpaceSortedUnitsSimulatedData(
            bandwidth: [5.0, 5.0],
            dimensions: 2,
            evaluationSteps: [50, 50],
            min: [0, 0],
            max: [100, 100],
            scale: 0.1,
            transitionsType: TransitionsType.RandomWalk,
            sigma: [0.1, 0.1],
            modelDirectory: "SimulatedData2D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelRandomWalkCompression2D()
    {
        TestModel.BayesianStateSpaceSortedUnitsSimulatedData(
            bandwidth: [5.0, 5.0],
            dimensions: 2,
            evaluationSteps: [50, 50],
            min: [0, 0],
            max: [100, 100],
            scale: 0.1,
            transitionsType: TransitionsType.RandomWalk,
            estimationMethod: EstimationMethod.KernelCompression,
            sigma: [0.1, 0.1],
            distanceThreshold: 1.5,
            modelDirectory: "SimulatedData2D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelUniformDensityRealData2D()
    {
        TestModel.BayesianStateSpaceSortedUnitsRealData(
            bandwidth: [0.5, 0.5],
            dimensions: 2,
            evaluationSteps: [50, 50],
            minVals: [0, 0],
            maxVals: [120, 120],
            testFraction: 0.01,
            trainingFraction: 0.2,
            modelDirectory: "RealData2D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelRandomWalkCompressionRealData2D()
    {
        TestModel.BayesianStateSpaceSortedUnitsRealData(
            bandwidth: [0.5, 0.5],
            dimensions: 2,
            evaluationSteps: [50, 50],
            minVals: [0, 0],
            maxVals: [120, 120],
            sigma: [0.25, 0.25],
            distanceThreshold: 1.5,
            testFraction: 0.01,
            trainingFraction: 0.2,
            modelDirectory: "RealData2D",
            transitionsType: TransitionsType.RandomWalk,
            estimationMethod: EstimationMethod.KernelCompression
        );
    }
}