using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Lenium.NeuralNetwork.TransferFunctions.Interfaces;

namespace Lenium.NeuralNetwork.TransferFunctions
{
    public class TanhTransferFunction : ITransferFunction
    {
        public double Compute(double x)
        {

            // return Math.Exp(-Math.Pow(x, 2));
            //    return Math.Tanh(x);
            //  return (2 / (1 + Math.Pow(Math.Exp(x), -2 * x))) - 1;
            //   return (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x));
            var alpha = 0.2;
            var sh = (Math.Round(Math.Exp(alpha * x), 5) - Math.Round(Math.Exp(-alpha * x), 5)) / 2;
            var ch = (Math.Round(Math.Exp(alpha * x), 5) + Math.Round(Math.Exp(-alpha * x), 5)) / 2;
            return sh / ch;
        }

        public double Derivative(double x)
        {
            var alpha = 0.2;
            return 2 * alpha * (x + 1) * (1.0 - (x + 1) / 2);
        }
    }
}
