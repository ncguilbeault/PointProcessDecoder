using static TorchSharp.torch;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Axes;

namespace PointProcessDecoder.Plot;

public class Heatmap : OxyPlotBase
{
    public override PlotModel Plot => _plot;
    private readonly PlotModel _plot;
    public double XMin { get; } = double.NaN;
    public double XMax { get; } = double.NaN;
    public double YMin { get; } = double.NaN;
    public double YMax { get; } = double.NaN;
    public double ValMin { get; } = double.NaN;
    public double ValMax { get; } = double.NaN;
    public string XAxisTitle { get; } = "X Axis";
    public string YAxisTitle { get; } = "Y Axis";
    public string Title { get; } = "Heatmap";

    public Heatmap()
    {
        FigureName = Title;

        _plot = new PlotModel 
        { 
            Title = Title,
            TitleFont = "DejaVu Sans",
            DefaultFont = "DejaVu Sans",
            Background = OxyColors.White
        };

        Initialize();
    }

    public Heatmap(
        double? xMin = null, 
        double? xMax = null, 
        double? yMin = null, 
        double? yMax = null, 
        double? valMin = null,
        double? valMax = null,
        string? xAxisTitle = null,
        string? yAxisTitle = null,
        string? title = null, 
        string? figureName = null
    )
    {
        XMin = xMin ?? XMin;
        XMax = xMax ?? XMax;
        YMin = yMin ?? YMin;
        YMax = yMax ?? YMax;
        ValMin = valMin ?? ValMin;
        ValMax = valMax ?? ValMax;
        XAxisTitle = xAxisTitle ?? XAxisTitle;
        YAxisTitle = yAxisTitle ?? YAxisTitle;
        Title = title ?? Title;
        FigureName = figureName ?? Title;

        _plot = new PlotModel 
        { 
            Title = Title,
            TitleFont = "DejaVu Sans",
            DefaultFont = "DejaVu Sans",
            Background = OxyColors.White
        };

        Initialize();
    }

    private void Initialize()
    {
        var xAxis = new LinearAxis 
        { 
            Position = AxisPosition.Bottom, 
            Title = XAxisTitle,
            TitleFont = "DejaVu Sans",
            Minimum = XMin,
            Maximum = XMax
        };

        var yAxis = new LinearAxis 
        { 
            Position = AxisPosition.Left, 
            Title = YAxisTitle,
            TitleFont = "DejaVu Sans",
            Minimum = YMin,
            Maximum = YMax
        };

        var colorAxis = new LinearColorAxis 
        { 
            Position = AxisPosition.Right, 
            Palette = OxyPalettes.Viridis(100), 
            TitleFont = "DejaVu Sans",
            Key = "color",
            Minimum = ValMin,
            Maximum = ValMax
        };

        var scatterColorAxis = new RangeColorAxis 
        { 
            Position = AxisPosition.None,
            Key = "scatterColor"
        };

        _plot.Axes.Add(colorAxis);
        _plot.Axes.Add(scatterColorAxis);
        _plot.Axes.Add(xAxis);
        _plot.Axes.Add(yAxis);
    }

    public void Show(Tensor density, Tensor? points = null, bool addLine = false)
    {
        var densityArray = density.to_type(ScalarType.Float64).data<double>().ToNDArray();
        var densityData = new double[density.shape[0], density.shape[1]];
        Array.Copy(densityArray, densityData, densityArray.Length);

        var heatMapSeries = new HeatMapSeries
        {
            X0 = XMin,
            X1 = XMax,
            Y0 = YMin,
            Y1 = YMax,
            Interpolate = true,
            Data = densityData,
            ColorAxisKey = "color"
        };

        _plot.Series.Add(heatMapSeries);

        if (points is not null)
        {
            AddPoints(points, addLine);
        }
    }

    public void Show(double[,] density)
    {
        var heatMapSeries = new HeatMapSeries
        {
            X0 = XMin,
            X1 = XMax,
            Y0 = YMin,
            Y1 = YMax,
            Interpolate = true,
            Data = density,
            ColorAxisKey = "color"
        };

        _plot.Series.Add(heatMapSeries);
    }

    private void AddPoints(Tensor points, bool addLine = false)
    {
        if (points.dim() != 2)
        {
            throw new ArgumentException("Points must have 2 dimensions.");
        }

        if (points.shape[1] != 2)
        {
            throw new ArgumentException("Points must have 2 columns.");
        }

        var scatterSeries = new ScatterSeries
        {
            MarkerType = MarkerType.Circle,
            MarkerFill = OxyColors.Red,
            MarkerSize = 4,
            MarkerStrokeThickness = 1,
            ColorAxisKey = "scatterColor"
        };

        var lineSeries = addLine ? new LineSeries
        {
            Color = OxyColors.Red,
            StrokeThickness = 1,
            MarkerType = MarkerType.None
        } : null;

        var pointsArray = points.to_type(ScalarType.Float64)
            .data<double>()
            .ToNDArray();

        var pointsData = new double[points.shape[0], points.shape[1]];
        Array.Copy(pointsArray, pointsData, pointsArray.Length);

        for (int i = 0; i < points.shape[0]; i++)
        {
            scatterSeries.Points.Add(new ScatterPoint(pointsData[i,0], pointsData[i,1], double.NaN, 0));
            lineSeries?.Points.Add(new DataPoint(pointsData[i,0], pointsData[i,1]));
        }
        _plot.Series.Add(scatterSeries);
        if (addLine)
        {
            _plot.Series.Add(lineSeries);
        }
    }
}