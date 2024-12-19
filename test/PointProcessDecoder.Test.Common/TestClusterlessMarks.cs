using static TorchSharp.torch;
using PointProcessDecoder.Plot;
using PointProcessDecoder.Core;
using PointProcessDecoder.Core.Estimation;
using PointProcessDecoder.Core.Transitions;
using PointProcessDecoder.Core.Encoder;
using PointProcessDecoder.Core.Decoder;
using PointProcessDecoder.Core.Likelihood;
using PointProcessDecoder.Core.StateSpace;
using PointProcessDecoder.Simulation;

namespace PointProcessDecoder.Test.Common;

public static class TestClusterlessMarks
{
    public static void BayesianStateSpaceClusterlessMarksSimulated(
        double[]? observationBandwidth = null,
        int dimensions = 1,
        long[]? evaluationSteps = null,
        int steps = 200,
        int cycles = 10,
        double[]? min = null,
        double[]? max = null,
        int numNeurons = 40,
        double placeFieldRadius = 8.0,
        double firingThreshold = 0.2,
        int markDimensions = 4,
        int markChannels = 8,
        double[]? markBandwidth = null,
        int nTraining = 1800,
        int nTesting = 200,
        double[]? sigma = null,
        double? distanceThreshold = null,
        string outputDirectory = "TestClusterlessMarks",
        string modelDirectory = "BayesianStateSpaceClusterlessMarksSimulatedData",
        ScalarType scalarType = ScalarType.Float32,
        Device? device = null,
        int seed = 0,
        double spikeScale = 5.0,
        double noiseScale = 0.5,
        string figureName = "Prediction",
        EstimationMethod estimationMethod = EstimationMethod.KernelDensity,
        TransitionsType transitionsType = TransitionsType.Uniform
    )
    {
        observationBandwidth ??= [5];
        evaluationSteps ??= [50];
        min ??= [0];
        max ??= [100];
        markBandwidth ??= [1, 1, 1, 1];
        device ??= CPU;

        outputDirectory = string.IsNullOrEmpty(modelDirectory) ? outputDirectory : Path.Combine(outputDirectory, modelDirectory);
        outputDirectory = Path.Combine(outputDirectory, estimationMethod.ToString(), transitionsType.ToString());
        
        var pointProcessModel = new PointProcessModel(
            estimationMethod,
            transitionsType,
            EncoderType.ClusterlessMarkEncoder,
            DecoderType.StateSpaceDecoder,
            StateSpaceType.DiscreteUniformStateSpace,
            LikelihoodType.Clusterless,
            min,
            max,
            evaluationSteps,
            observationBandwidth,
            dimensions,
            markDimensions: markDimensions,
            markChannels: markChannels,
            markBandwidth: markBandwidth,
            nUnits: numNeurons,
            distanceThreshold: distanceThreshold,
            sigmaRandomWalk: sigma,
            device: device,
            scalarType: scalarType
        );

        Tensor position = empty(0);
        Tensor spikingData = empty(0);
        Tensor positionPoints = empty(0);
        double[] heatmapRange = new double[4];

        if (dimensions == 1)
        {
            (position, spikingData) = Utilities.InitializeSimulation1D(
                steps: steps,
                cycles: cycles,
                min: min[0],
                max: max[0],
                numNeurons: numNeurons,
                placeFieldRadius: placeFieldRadius,
                firingThreshold: firingThreshold,
                scalarType: scalarType,
                seed: seed
            );

            positionPoints = concat([arange(steps * cycles - nTraining).unsqueeze(-1), position[TensorIndex.Slice(nTraining, nTraining + nTesting)]], dim: 1);
            heatmapRange = [0, steps * cycles - nTraining, min[0], max[0]];
        }
        else if (dimensions == 2)
        {
            (position, spikingData) = Utilities.InitializeSimulation2D(
                steps: steps,
                cycles: cycles,
                xMin: min[0],
                xMax: max[0],
                yMin: min[1],
                yMax: max[1],
                numNeurons: numNeurons,
                placeFieldRadius: placeFieldRadius,
                firingThreshold: firingThreshold,
                scalarType: scalarType,
                seed: seed
            );

            positionPoints = position[TensorIndex.Slice(nTraining, nTraining + nTesting)];
            heatmapRange = [min[0], max[0], min[1], max[1]];
        }
        
        var marks = Simulate.MarksAtPosition(
            position,
            spikingData, 
            markDimensions, 
            markChannels, 
            scalarType: scalarType, 
            device: device,
            spikeScale: spikeScale,
            noiseScale: noiseScale
        );

        pointProcessModel.Encode(position[TensorIndex.Slice(0, nTraining)], marks[TensorIndex.Slice(0, nTraining)]);
        var prediction = pointProcessModel.Decode(marks[TensorIndex.Slice(nTraining, nTraining + nTesting)]);
        if (dimensions == 2)
            prediction = prediction.sum(dim: 0);

        Heatmap plotPrediction = new(
            heatmapRange[0],
            heatmapRange[1],
            heatmapRange[2],
            heatmapRange[3],
            title: figureName
        );

        plotPrediction.OutputDirectory = Path.Combine(plotPrediction.OutputDirectory, outputDirectory);
        plotPrediction.Show<float>(
            prediction, 
            positionPoints
        );
        plotPrediction.Save(png: true);
    }
}