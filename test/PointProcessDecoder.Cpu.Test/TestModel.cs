using PointProcessDecoder.Core.Transitions;
using PointProcessDecoder.Core.Estimation;
using PointProcessDecoder.Test.Common;

namespace PointProcessDecoder.Cpu.Test;

[TestClass]
public class TestModel
{
    [TestMethod]
    public void TestPointProcessModelSortedUnitsUniformDensitySimulatedData()
    {
        SortedUnitsUtilities.BayesianStateSpaceSortedUnitsSimulatedData();
    }

    [TestMethod]
    public void TestPointProcessModelSortedUnitsRandomWalkDensitySimulatedData()
    {
        SortedUnitsUtilities.BayesianStateSpaceSortedUnitsSimulatedData(
            transitionsType: TransitionsType.RandomWalk,
            sigma: 0.1,
            modelDirectory: "SimulatedData1D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelSortedUnitsRandomWalkCompressionSimulatedData()
    {
        SortedUnitsUtilities.BayesianStateSpaceSortedUnitsSimulatedData(
            transitionsType: TransitionsType.RandomWalk,
            estimationMethod: EstimationMethod.KernelCompression,
            sigma: 0.1,
            distanceThreshold: 1.5,
            modelDirectory: "SimulatedData1D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelSortedUnitsUniformDensitySimulatedData2D()
    {
        SortedUnitsUtilities.BayesianStateSpaceSortedUnitsSimulatedData(
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
    public void TestPointProcessModelSortedUnitsRandomWalkDensitySimulatedData2D()
    {
        SortedUnitsUtilities.BayesianStateSpaceSortedUnitsSimulatedData(
            bandwidth: [5.0, 5.0],
            dimensions: 2,
            evaluationSteps: [50, 50],
            min: [0, 0],
            max: [100, 100],
            scale: 0.1,
            transitionsType: TransitionsType.RandomWalk,
            sigma: 0.1,
            modelDirectory: "SimulatedData2D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelSortedUnitsRandomWalkCompressionSimulatedData2D()
    {
        SortedUnitsUtilities.BayesianStateSpaceSortedUnitsSimulatedData(
            bandwidth: [5.0, 5.0],
            dimensions: 2,
            evaluationSteps: [50, 50],
            min: [0, 0],
            max: [100, 100],
            scale: 0.1,
            transitionsType: TransitionsType.RandomWalk,
            estimationMethod: EstimationMethod.KernelCompression,
            sigma: 0.1,
            distanceThreshold: 1.5,
            modelDirectory: "SimulatedData2D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelSortedUnitsUniformDensityRealData2D()
    {
        SortedUnitsUtilities.BayesianStateSpaceSortedUnitsRealData(
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
    public void TestPointProcessModelSortedUnitsRandomWalkCompressionRealData2D()
    {
        SortedUnitsUtilities.BayesianStateSpaceSortedUnitsRealData(
            bandwidth: [0.5, 0.5],
            dimensions: 2,
            evaluationSteps: [50, 50],
            minVals: [0, 0],
            maxVals: [120, 120],
            sigma: 0.25,
            distanceThreshold: 1.5,
            testFraction: 0.01,
            trainingFraction: 0.2,
            modelDirectory: "RealData2D",
            transitionsType: TransitionsType.RandomWalk,
            estimationMethod: EstimationMethod.KernelCompression
        );
    }

    [TestMethod]
    public void TestPointProcessModelClusterlessMarksUniformDensitySimulatedData()
    {
        ClusterlessMarksUtilities.BayesianStateSpaceClusterlessMarksSimulated(
            transitionsType: TransitionsType.Uniform,
            observationBandwidth: [1],
            markBandwidth: [0.5,0.5,0.5,0.5],
            firingThreshold: 0.5,
            noiseScale: 0.5,
            modelDirectory: "SimulatedData"
        );
    }

    [TestMethod]
    public void TestPointProcessModelClusterlessMarksUniformCompressionSimulatedData()
    {
        ClusterlessMarksUtilities.BayesianStateSpaceClusterlessMarksSimulated(
            transitionsType: TransitionsType.Uniform,
            estimationMethod: EstimationMethod.KernelCompression,
            distanceThreshold: 1.5,
            observationBandwidth: [2],
            markBandwidth: [1,1,1,1],
            firingThreshold: 0.2,
            noiseScale: 1.0,
            modelDirectory: "SimulatedData"
        );
    }

    [TestMethod]
    public void TestPointProcessModelClusterlessMarksUniformDensitySimulatedData2D()
    {
        ClusterlessMarksUtilities.BayesianStateSpaceClusterlessMarksSimulated(
            transitionsType: TransitionsType.Uniform,
            observationBandwidth: [5,5],
            dimensions : 2,
            evaluationSteps: [50, 50],
            min: [0, 0],
            max: [100, 100],
            scale: 0.1,
            markBandwidth: [0.5,0.5,0.5,0.5],
            firingThreshold: 0.4,
            noiseScale: 0.5,
            modelDirectory: "SimulatedData2D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelClusterlessMarksUniformCompressionSimulatedData2D()
    {
        ClusterlessMarksUtilities.BayesianStateSpaceClusterlessMarksSimulated(
            transitionsType: TransitionsType.Uniform,
            estimationMethod: EstimationMethod.KernelCompression,
            distanceThreshold: 1.5,
            observationBandwidth: [5,5],
            dimensions : 2,
            evaluationSteps: [50, 50],
            min: [0, 0],
            max: [100, 100],
            scale: 0.1,
            markBandwidth: [0.5,0.5,0.5,0.5],
            firingThreshold: 0.4,
            noiseScale: 0.5,
            modelDirectory: "SimulatedData2D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelClusterlessMarksRandomWalkDensitySimulatedData()
    {
        ClusterlessMarksUtilities.BayesianStateSpaceClusterlessMarksSimulated(
            transitionsType: TransitionsType.RandomWalk,
            observationBandwidth: [5],
            sigma: 3,
            markBandwidth: [2,2,2,2],
            firingThreshold: 0.5,
            noiseScale: 0.5,
            modelDirectory: "SimulatedData"
        );
    }

    [TestMethod]
    public void TestPointProcessModelClusterlessMarksRandomWalkCompressionSimulatedData()
    {
        ClusterlessMarksUtilities.BayesianStateSpaceClusterlessMarksSimulated(
            transitionsType: TransitionsType.RandomWalk,
            estimationMethod: EstimationMethod.KernelCompression,
            distanceThreshold: 1.5,
            observationBandwidth: [5],
            sigma: 3,
            markBandwidth: [2,2,2,2],
            firingThreshold: 0.5,
            noiseScale: 0.5,
            modelDirectory: "SimulatedData"
        );
    }

    [TestMethod]
    public void TestPointProcessModelClusterlessMarksRandomWalkDensitySimulatedData2D()
    {
        ClusterlessMarksUtilities.BayesianStateSpaceClusterlessMarksSimulated(
            transitionsType: TransitionsType.RandomWalk,
            observationBandwidth: [5, 5],
            sigma: 1,
            dimensions : 2,
            evaluationSteps: [50, 50],
            min: [0, 0],
            max: [100, 100],
            scale: 0.1,
            markBandwidth: [0.5,0.5,0.5,0.5],
            firingThreshold: 0.4,
            noiseScale: 0.5,
            modelDirectory: "SimulatedData2D"
        );
    }

    [TestMethod]
    public void TestPointProcessModelClusterlessMarksRandomWalkCompressionSimulatedData2D()
    {
        ClusterlessMarksUtilities.BayesianStateSpaceClusterlessMarksSimulated(
            transitionsType: TransitionsType.RandomWalk,
            estimationMethod: EstimationMethod.KernelCompression,
            distanceThreshold: 1,
            observationBandwidth: [10, 10],
            sigma: 10,
            dimensions : 2,
            evaluationSteps: [50, 50],
            min: [0, 0],
            max: [100, 100],
            scale: 0.1,
            markBandwidth: [2,2,2,2],
            firingThreshold: 0.4,
            noiseScale: 0.5,
            modelDirectory: "SimulatedData2D"
        );
    }
}