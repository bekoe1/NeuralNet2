using System;
using System.Drawing;
using System.Drawing.Drawing2D;

namespace NeuralNetwork1
{
    public static class AugmentationHelper
    {
        private static Random random = new Random();

        /// <summary>
        /// Генерирует пачку искаженных копий из одного образца
        /// </summary>
        /// <param name="baseSample">Исходный образец</param>
        /// <param name="count">Сколько копий создать</param>
        /// <returns>Массив новых сэмплов</returns>
        public static Sample[] GenerateSeries(Sample baseSample, int count)
        {
            Sample[] result = new Sample[count];
            int width = 32; // Размер картинки в твоем проекте
            int height = 32;

            // Конвертируем исходный вектор обратно в картинку для обработки
            using (Bitmap originalBmp = ArrayToBitmap(baseSample.input, width, height))
            {
                for (int i = 0; i < count; i++)
                {
                    // Создаем искаженную копию
                    double[] augmentedInput = ProcessBitmap(originalBmp, width, height);
                    result[i] = new Sample(augmentedInput, 10, baseSample.actualClass);
                }
            }
            return result;
        }

        private static double[] ProcessBitmap(Bitmap original, int w, int h)
        {
            using (Bitmap transformed = new Bitmap(w, h))
            using (Graphics g = Graphics.FromImage(transformed))
            {
                // 1. Настройка графики (как в примере)
                g.Clear(Color.Black); // Фон черный, цифра белая (так удобнее для математики)
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.SmoothingMode = SmoothingMode.HighQuality;

                // 2. Параметры искажений (как в ImageGenerator.cs)
                float angle = (float)(random.NextDouble() * 30.0 - 15.0); // Поворот +/- 15
                float scale = (float)(0.85 + random.NextDouble() * 0.3);  // Масштаб 0.85 - 1.15
                float shiftX = (float)(random.NextDouble() * 6.0 - 3.0);  // Сдвиг +/- 3 пикселя
                float shiftY = (float)(random.NextDouble() * 6.0 - 3.0);

                // 3. Применяем матрицу трансформации
                Matrix mat = new Matrix();
                mat.Translate(w / 2f, h / 2f);      // В центр
                mat.Rotate(angle);                  // Крутим
                mat.Scale(scale, scale);            // Масштабируем
                mat.Translate(-w / 2f, -h / 2f);    // Обратно
                mat.Translate(shiftX, shiftY);      // Сдвигаем

                g.Transform = mat;
                g.DrawImage(original, 0, 0);

                // 4. Эффекты (шум и толщина) - применяем уже к пикселям
                return BitmapToInputVectorWithNoise(transformed);
            }
        }

        private static Bitmap ArrayToBitmap(double[] input, int width, int height)
        {
            Bitmap bmp = new Bitmap(width, height);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // В input: 1.0 - чернила, 0.0 - фон.
                    // Для GDI: 255 - белый (рисунок), 0 - черный (фон)
                    int val = (int)(input[y * width + x] * 255.0);
                    val = Math.Max(0, Math.Min(255, val));
                    bmp.SetPixel(x, y, Color.FromArgb(val, val, val));
                }
            }
            return bmp;
        }

        private static double[] BitmapToInputVectorWithNoise(Bitmap bmp)
        {
            int w = bmp.Width;
            int h = bmp.Height;
            double[] result = new double[w * h];

            // Решаем, будем ли менять толщину (Эрозия/Дилатация)
            // 30% шанс утолщения, 15% шанс утоньшения
            int morphType = 0; // 0 - ничего, 1 - dilate (толще), -1 - erode (тоньше)
            double r = random.NextDouble();
            if (r < 0.30) morphType = 1;
            else if (r > 0.85) morphType = -1;

            bool addNoise = random.NextDouble() < 0.5; // 50% шанс шума

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    double val = 0;

                    // Логика морфологии (упрощенная)
                    if (morphType == 0)
                    {
                        val = bmp.GetPixel(x, y).R / 255.0;
                    }
                    else if (morphType == 1) // Утолщение (ищем яркого соседа)
                    {
                        val = MaxNeighbor(bmp, x, y);
                    }
                    else // Утоньшение (ищем темного соседа)
                    {
                        val = MinNeighbor(bmp, x, y);
                    }

                    // Добавляем шум (Jitter)
                    if (addNoise && random.NextDouble() < 0.05) // 5% пикселей шумят
                    {
                        val += (random.NextDouble() - 0.5) * 0.4;
                    }

                    result[y * w + x] = Math.Max(0, Math.Min(1, val));
                }
            }
            return result;
        }

        // Вспомогательные методы для морфологии
        private static double MaxNeighbor(Bitmap b, int x, int y)
        {
            double max = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    int nx = x + dx, ny = y + dy;
                    if (nx >= 0 && nx < b.Width && ny >= 0 && ny < b.Height)
                        max = Math.Max(max, b.GetPixel(nx, ny).R / 255.0);
                }
            return max;
        }

        private static double MinNeighbor(Bitmap b, int x, int y)
        {
            double min = 1.0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    int nx = x + dx, ny = y + dy;
                    if (nx >= 0 && nx < b.Width && ny >= 0 && ny < b.Height)
                        min = Math.Min(min, b.GetPixel(nx, ny).R / 255.0);
                    else
                        min = 0; // За границей считаем фон
                }
            return min;
        }
    }
}