using System;
using System.Drawing;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace NeuralNetwork1
{
    public class CameraProcessor : IDisposable
    {
        private VideoCapture _capture;
        private Mat _frame;
        private bool _isCapturing;

        // Итоговый размер входа нейросети
        private const int NetworkInputSize = 32;

        // Поле для отладки (показывает, что видит сеть)
        public Mat ProcessedView { get; private set; }

        public CameraProcessor()
        {
            _frame = new Mat();
            ProcessedView = new Mat();
        }

        public void Start()
        {
            if (_capture == null)
                _capture = new VideoCapture(0);

            if (!_capture.IsOpened())
                throw new Exception("Не удалось подключиться к камере!");

            _isCapturing = true;
        }

        public void Stop()
        {
            _isCapturing = false;
            _capture?.Release();
            _capture = null;
        }

        /// <summary>
        /// Возвращает область "прицела" (центр экрана)
        /// </summary>
        private Rect GetScanArea(int frameWidth, int frameHeight)
        {
            // Размер квадрата сканирования (например, 300x300 пикселей)
            int size = Math.Min(frameWidth, frameHeight) / 2;
            int x = (frameWidth - size) / 2;
            int y = (frameHeight - size) / 2;
            return new Rect(x, y, size, size);
        }

        public Bitmap GetFrame()
        {
            if (!_isCapturing || _capture == null) return null;

            _capture.Read(_frame);
            if (_frame.Empty()) return null;

            Rect scanZone = GetScanArea(_frame.Width, _frame.Height);
            Cv2.Rectangle(_frame, scanZone, Scalar.Blue, 2);

            using (Mat crop = new Mat(_frame, scanZone)) // Работаем только с центром
            {
                Rect digitRect = FindDigitBoundingBox(crop);
                if (digitRect.Width > 0)
                {
                    digitRect.X += scanZone.X;
                    digitRect.Y += scanZone.Y;
                    Cv2.Rectangle(_frame, digitRect, Scalar.LightGreen, 2);
                }
            }

            if (!ProcessedView.Empty())
            {
                try
                {
                    using (Mat debugMini = new Mat())
                    using (Mat debugColor = new Mat())
                    {
                        Cv2.Resize(ProcessedView, debugMini, new OpenCvSharp.Size(100, 100), 0, 0, InterpolationFlags.Nearest);
                        Cv2.CvtColor(debugMini, debugColor, ColorConversionCodes.GRAY2BGR);
                        var roi = new Rect(0, 0, 100, 100);
                        debugColor.CopyTo(_frame[roi]);
                        Cv2.Rectangle(_frame, roi, Scalar.Yellow, 2);
                    }
                }
                catch { }
            }

            return BitmapConverter.ToBitmap(_frame);
        }

        public Sample ProcessCurrentFrame(FigureType type = FigureType.Undef)
        {
            if (_frame.Empty()) return null;

            Rect scanZone = GetScanArea(_frame.Width, _frame.Height);

            using (Mat crop = new Mat(_frame, scanZone))
            {
                Rect rect = FindDigitBoundingBox(crop);

                if (rect.Width <= 5 || rect.Height <= 5) return null;

                using (Mat digitROI = new Mat(crop, rect))
                using (Mat gray = new Mat())
                using (Mat binary = new Mat())
                {
                    Cv2.CvtColor(digitROI, gray, ColorConversionCodes.BGR2GRAY);

                    Cv2.Threshold(gray, binary, 0, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu);

                    var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(3, 3));
                    Cv2.Dilate(binary, binary, kernel);

                    using (Mat finalImage = Mat.Zeros(NetworkInputSize, NetworkInputSize, MatType.CV_8UC1))
                    {
                        int targetSize = 22; 
                        double scale = (double)targetSize / Math.Max(binary.Width, binary.Height);
                        int newW = (int)(binary.Width * scale);
                        int newH = (int)(binary.Height * scale);
                        newW = Math.Max(1, newW);
                        newH = Math.Max(1, newH);

                        using (Mat resizedDigit = new Mat())
                        {
                            Cv2.Resize(binary, resizedDigit, new OpenCvSharp.Size(newW, newH), 0, 0, InterpolationFlags.Area);

                            int offsetX = (NetworkInputSize - newW) / 2;
                            int offsetY = (NetworkInputSize - newH) / 2;

                            var targetRegion = new Rect(offsetX, offsetY, newW, newH);
                            resizedDigit.CopyTo(finalImage[targetRegion]);
                        }

                        finalImage.CopyTo(ProcessedView);

                        double[] input = new double[NetworkInputSize * NetworkInputSize];
                        for (int y = 0; y < NetworkInputSize; y++)
                        {
                            for (int x = 0; x < NetworkInputSize; x++)
                            {
                                input[y * NetworkInputSize + x] = finalImage.At<byte>(y, x) > 10 ? 1.0 : 0.0;
                            }
                        }
                        return new Sample(input, 10, type);
                    }
                }
            }
        }

        private Rect FindDigitBoundingBox(Mat src)
        {
            using (Mat gray = new Mat())
            using (Mat blurred = new Mat())
            using (Mat binary = new Mat())
            {
                Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);
                Cv2.GaussianBlur(gray, blurred, new OpenCvSharp.Size(5, 5), 0);
                Cv2.Threshold(blurred, binary, 0, 255, ThresholdTypes.BinaryInv | ThresholdTypes.Otsu);

                OpenCvSharp.Point[][] contours;
                HierarchyIndex[] hierarchy;
                Cv2.FindContours(binary, out contours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

                if (contours.Length == 0) return new Rect(0, 0, 0, 0);

                var maxContour = contours.OrderByDescending(c => Cv2.ContourArea(c)).First();

                // Фильтр шума: объект должен быть достаточно большим
                if (Cv2.ContourArea(maxContour) < 50) return new Rect(0, 0, 0, 0);

                Rect r = Cv2.BoundingRect(maxContour);

                // Добавляем поля
                int pad = 5;
                r.X = Math.Max(0, r.X - pad);
                r.Y = Math.Max(0, r.Y - pad);
                r.Width = Math.Min(src.Width - r.X, r.Width + pad * 2);
                r.Height = Math.Min(src.Height - r.Y, r.Height + pad * 2);

                return r;
            }
        }

        public void Dispose()
        {
            Stop();
            _frame?.Dispose();
            ProcessedView?.Dispose();
        }
    }
}