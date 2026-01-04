using System;
using System.Drawing;

namespace NeuralNetwork1
{
    public enum FigureType : byte
    {
        Zero = 0, One, Two, Three, Four, Five, Six, Seven, Eight, Nine, Undef
    };

    public class GenerateImage
    {
        public const int ImageResolution = 32;
        public int FigureCount { get; set; } = 10;

        public Bitmap GenBitmap() { return new Bitmap(200, 200); }
    }
}