import torch
from torch import nn
import math

class STE_Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Because we are saving one of the inputs use `save_for_backward`
        # Save non-tensors and non-inputs/non-outputs directly on ctx
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_out):
        # A function support double backward automatically if autograd
        # is able to record the computations performed in backward
        return grad_out

def Q(x, s, z, alpha_q, beta_q):
  # WARNING TORCH.ROUND BACKPROP GIVES 0, SHOULD USE STE
  x_q = STE_Round.apply(1/s*x+z)
  #x_q = 1/s * x + z
  return torch.clamp(x_q, min=alpha_q, max=beta_q)

def Q_int8(x, s, z):
  x_q = Q(x, s, z, alpha_q = -127, beta_q = 127)
  return x_q

def Q_uint8(x, s, z):
  x_q = Q(x, s, z, alpha_q = 0, beta_q = 255)
  return x_q

def Q_inv(x_q, s, z):
  return s * (x_q - z)

def Q_matmul_s_only(x, w, s_x, s_w):
  return (x @ w) * s_x * s_w

def FQ_int8(x, s, z):
  return Q_inv(Q_int8(x, s, z), s, z)

def FQ(x, s, z, bins):
  return Q_inv(Q(x, s, z, alpha_q = -bins, beta_q = bins), s, z)

class t(nn.Module):
  def __init__(self, x_size, y_size):
    super().__init__()
    self.weights = nn.Parameter(torch.randn(x_size, y_size) / math.sqrt(x_size))
    self.bias = nn.Parameter(torch.zeros(y_size))
    self.s_x = lambda bits: 2*self.max_x/bits
    self.max_x = 0.0
    self.s_w = lambda bits: 2*torch.max(torch.abs(self.weights))/bits
    self.s_b = lambda bits: 2*torch.max(torch.abs(self.bias))/bits

  def forward(self, xb, train = True, w_bits = 256, x_bits = 256):
    if train:
      # QAT for weight only, find the max range of input over training
      curr = torch.max(torch.abs(xb))
      self.max_x = curr if (curr > self.max_x) else self.max_x
      return (xb @ FQ_int8(self.weights, self.s_w(w_bits), 0)) + self.bias
    else:
      # Use int8
      Q_xb = Q_int8(xb, self.s_x(x_bits), 0).type(torch.int8)
      # TODO: DO NOT CONVERT IT TO FLOAT, USE FIXED POINT TO ADD BIAS AND DEQUANTIZE
      # https://arxiv.org/pdf/1712.05877.pdf see eq 5 & 6
      return (Q_xb @ self.Q_w(w_bits)).type(torch.float) * self.s_x(x_bits) * self.s_w(w_bits) + self.Q_b(x_bits) * self.s_b(x_bits)

  def Q_w(self, bits):
    return Q_int8(self.weights, self.s_w(bits), 0).type(torch.int8)

  def Q_b(self, bits):
    return Q_int8(self.bias, self.s_b(bits), 0).type(torch.int8)