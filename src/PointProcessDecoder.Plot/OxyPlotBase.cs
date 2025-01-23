using OxyPlot;
using OxyPlot.SkiaSharp;

namespace PointProcessDecoder.Plot;

public abstract class OxyPlotBase
{
    public string FigureName { get; set; } = "";
    public string OutputDirectory { get; set; } = "figures";
    public bool SavePng { get; set; } = true;
    public bool SavePdf { get; set; } = true;
    public bool SaveSvg { get; set; } = true;
    public int Width { get; set; } = 600;
    public int Height { get; set; } = 600;
    public abstract PlotModel Plot { get; }

    public void Save(bool png = false, bool pdf = false, bool svg = false)
    {
        SavePng = png;
        SavePdf = pdf;
        SaveSvg = svg;
        Save();
    }

    public void Save()
    {
        if (!SavePng && !SavePdf && !SaveSvg)
        {
            return;
        }

        if (!Directory.Exists(OutputDirectory))
        {
            Directory.CreateDirectory(OutputDirectory);
        }

        if (SavePng)
        {
            SavePlotAsPng();
        }

        if (SavePdf)
        {
            SavePlotAsPdf();
        }

        if (SaveSvg)
        {
            SavePlotAsSvg();
        }
    }

    public void SavePlotAsPng()
    {
        var exporter = new PngExporter() {Width = Width, Height = Height};
        var path = Path.Combine(OutputDirectory, $"{FigureName}.png");
        using var stream = File.Create(path);
        exporter.Export(Plot, stream);
    }

    public void SavePlotAsPdf()
    {
        var exporter = new OxyPlot.SkiaSharp.PdfExporter() {Width = 600, Height = 600};
        var path = Path.Combine(OutputDirectory, $"{FigureName}.pdf");
        using var stream = File.Create(path);
        exporter.Export(Plot, stream);
    }

    public void SavePlotAsSvg()
    {
        var exporter = new OxyPlot.SkiaSharp.SvgExporter() {Width = 600, Height = 600};
        var path = Path.Combine(OutputDirectory, $"{FigureName}.svg");
        using var stream = File.Create(path);
        exporter.Export(Plot, stream);
    }

    public byte[] ToBytes()
    {
        var exporter = new PngExporter() {Width = Width, Height = Height};
        using var stream = new MemoryStream();
        exporter.Export(Plot, stream);
        return stream.ToArray();
    }
}