using static TorchSharp.torch;
using OxyPlot;
using OxyPlot.Series;
using OxyPlot.Axes;
using TorchSharp;

namespace PointProcessDecoder.Plot;

public class ScatterPlot : OxyPlotBase
{
    public override PlotModel Plot => _plot;
    private readonly PlotModel _plot;
    public double XMin { get; } = 0;
    public double XMax { get; } = 100;
    public double YMin { get; } = 0;
    public double YMax { get; } = 100;
    public string XAxisTitle { get; } = "X Axis";
    public string YAxisTitle { get; } = "Y Axis";
    public string Title { get; } = "ScatterPlot";

    public ScatterPlot()
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

    public ScatterPlot(
        double? xMin = null, 
        double? xMax = null, 
        double? yMin = null, 
        double? yMax = null, 
        string? xAxisTitle = null,
        string? yAxisTitle = null,
        string? title = null, 
        string? figureName = null,
        bool logX = false,
        bool logY = false
    )
    {
        XMin = xMin ?? XMin;
        XMax = xMax ?? XMax;
        YMin = yMin ?? YMin;
        YMax = yMax ?? YMax;
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

        Initialize(logX, logY);
    }

    private void Initialize(bool logX = false, bool logY = false)
    {
        Axis xAxis = logX ? new LogarithmicAxis 
        { 
            Position = AxisPosition.Bottom, 
            Title = XAxisTitle,
            TitleFont = "DejaVu Sans",
            Minimum = XMin,
            Maximum = XMax
        } : new LinearAxis 
        { 
            Position = AxisPosition.Bottom, 
            Title = XAxisTitle,
            TitleFont = "DejaVu Sans",
            Minimum = XMin,
            Maximum = XMax
        };

        Axis yAxis = logY ? new LogarithmicAxis 
        { 
            Position = AxisPosition.Left, 
            Title = YAxisTitle,
            TitleFont = "DejaVu Sans",
            Minimum = YMin,
            Maximum = YMax
        } : new LinearAxis 
        { 
            Position = AxisPosition.Left, 
            Title = YAxisTitle,
            TitleFont = "DejaVu Sans",
            Minimum = YMin,
            Maximum = YMax
        };

        _plot.Axes.Add(xAxis);
        _plot.Axes.Add(yAxis);
    }

    public void Show(
        Tensor data, 
        OxyColor? color = null, 
        bool addLine = false,
        string seriesLabel = ""
    )
    {
        data = data.to_type(ScalarType.Float64);

        var scatterSeries = new ScatterSeries
        {
            MarkerType = MarkerType.Circle,
            MarkerFill = color ?? OxyColors.Red,
            MarkerSize = 4,
            MarkerStrokeThickness = 1,
        };
        
        var lineSeries = addLine ? new LineSeries
        {
            Color = color ?? OxyColors.Red,
            StrokeThickness = 1,
            MarkerType = MarkerType.None
        } : null;

        if (!string.IsNullOrEmpty(seriesLabel))
        {
            if (lineSeries != null)
                lineSeries.LegendKey = seriesLabel;
            else
                scatterSeries.LegendKey = seriesLabel;

            _plot.IsLegendVisible = true;
        }
        
        for (int i = 0; i < data.shape[0]; i++)
        {
            scatterSeries.Points.Add(new ScatterPoint(
                data[i,0].item<double>(), 
                data[i,1].item<double>()
            ));
            lineSeries?.Points.Add(new DataPoint(
                data[i,0].item<double>(), 
                data[i,1].item<double>()
            ));
        }

        _plot.Series.Add(scatterSeries);
        if (addLine)
        {
            _plot.Series.Add(lineSeries);
        }
    }
}