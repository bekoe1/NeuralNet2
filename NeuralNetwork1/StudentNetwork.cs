using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        private List<double[,]> prevWeightsDelta;

        private List<double[,]> weights;
        private List<double[]> biases;
        private List<double[]> layers;
        private Random random;
        private double learningRate = 0.25;

        public StudentNetwork(int[] structure)
        {
            random = new Random();
            weights = new List<double[,]>();
            biases = new List<double[]>();
            layers = new List<double[]>();

            for (int i = 0; i < structure.Length; i++)
            {
                layers.Add(new double[structure[i]]);
            }

            for (int i = 0; i < structure.Length - 1; i++)
            {
                int rows = structure[i];
                int cols = structure[i + 1];

                double[,] w = new double[rows, cols];
                double[] b = new double[cols];

                for (int c = 0; c < cols; c++)
                {
                    b[c] = (random.NextDouble() * 2 - 1) * 0.5;
                    for (int r = 0; r < rows; r++)
                    {
                        w[r, c] = (random.NextDouble() * 2 - 1) * 0.5;
                    }
                }

                weights.Add(w);
                biases.Add(b);
            }
        }

        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private double SigmoidDerivative(double x)
        {
            return x * (1.0 - x);
        }

        protected override double[] Compute(double[] input)
        {
            for (int i = 0; i < input.Length; i++)
            {
                layers[0][i] = input[i];
            }

            for (int i = 0; i < weights.Count; i++)
            {
                for (int j = 0; j < layers[i + 1].Length; j++)
                {
                    double sum = biases[i][j];
                    for (int k = 0; k < layers[i].Length; k++)
                    {
                        sum += layers[i][k] * weights[i][k, j];
                    }
                    layers[i + 1][j] = Sigmoid(sum);
                }
            }

            return layers.Last();
        }

        private void BackPropagation(double[] expectedOutput)
        {
            var outputLayer = layers.Last();
            var deltas = new List<double[]>();

            double[] outputDeltas = new double[outputLayer.Length];
            for (int i = 0; i < outputLayer.Length; i++)
            {
                double error = expectedOutput[i] - outputLayer[i];
                outputDeltas[i] = error * SigmoidDerivative(outputLayer[i]);
            }
            deltas.Add(outputDeltas);

            for (int i = weights.Count - 1; i > 0; i--)
            {
                double[] prevDeltas = deltas.Last();
                double[] currentDeltas = new double[layers[i].Length];

                for (int j = 0; j < layers[i].Length; j++)
                {
                    double error = 0.0;
                    for (int k = 0; k < prevDeltas.Length; k++)
                    {
                        error += prevDeltas[k] * weights[i][j, k];
                    }
                    currentDeltas[j] = error * SigmoidDerivative(layers[i][j]);
                }
                deltas.Add(currentDeltas);
            }

            deltas.Reverse();

            for (int i = 0; i < weights.Count; i++)
            {
                double[] layerDeltas = deltas[i];
                double[] layerInputs = layers[i];

                for (int j = 0; j < layerDeltas.Length; j++)
                {
                    biases[i][j] += learningRate * layerDeltas[j];
                    for (int k = 0; k < layerInputs.Length; k++)
                    {
                        weights[i][k, j] += learningRate * layerDeltas[j] * layerInputs[k];
                    }
                }
            }
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int iterations = 0;
            while (iterations < 1000)
            {
                double[] output = Compute(sample.input);
                double error = 0.0;

                for (int i = 0; i < output.Length; i++)
                {
                    double diff = sample.Output[i] - output[i];
                    error += diff * diff;
                }

                if (error < acceptableError) break;

                BackPropagation(sample.Output);
                iterations++;
            }
            return iterations;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            double error = double.MaxValue;
            int epoch = 0;

            while (epoch < epochsCount && error > acceptableError)
            {
                error = 0.0;

                for (int i = 0; i < samplesSet.Count; i++)
                {
                    Sample sample = samplesSet[i];
                    double[] output = Compute(sample.input);

                    double sampleError = 0.0;
                    for (int j = 0; j < output.Length; j++)
                    {
                        double diff = sample.Output[j] - output[j];
                        sampleError += diff * diff;
                    }
                    error += sampleError;

                    BackPropagation(sample.Output);
                }

                error /= samplesSet.Count;
                OnTrainProgress((double)epoch / epochsCount, error, TimeSpan.Zero);
                epoch++;
            }

            OnTrainProgress(1.0, error, TimeSpan.Zero);
            return error;
        }
    }
}