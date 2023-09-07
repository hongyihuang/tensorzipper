import struct
import io

from pysrc import zipper

# OPCODE, 32 max currently
OP_BEGIN = 0
OP_END = 1
OP_MATMUL = 2

OP_MATRIX_MAX = 4
OP_MATRIX_ADD = 5

OP_QUANTIZE = 8
OP_DEQUANTIZE = 9

FILE_VERSION = 0

class op:
  def __init__(self, opcode, data, prev = None):
    self.opcode = opcode

    self.prev = prev
    self.next = None

    # Initialize buffer if there are no previous links (a begin node)
    if prev:
      (self.op_buf, self.weight_buf, self.data_buf) = prev.link_next(self)
    else:
      self.op_buf = io.BytesIO()
      self.weight_buf = io.BytesIO()
      self.data_buf = io.BytesIO()

    # Create tensor object to emit different data and weight bitstreams
    if (opcode == OP_BEGIN) or (opcode == OP_END):
      self.tensor = tensor(data, self.data_buf)
    elif (opcode == OP_QUANTIZE) or (opcode == OP_DEQUANTIZE):
      self.tensor = data
    else:
      self.tensor = tensor(data, self.weight_buf)

  def emit(self):
    self.op_buf.write(self.opcode.to_bytes(1, 'little'))

    if (self.opcode == OP_QUANTIZE) or (self.opcode == OP_DEQUANTIZE):
      # directly store scalar
      (tmp) = struct.pack('<f', self.tensor) # actually a float32
      self.op_buf.write(tmp[1:3])
    else:
      self.op_buf.write(self.tensor.buf.tell().to_bytes(3, 'little'))
      self.tensor.emit()
    if self.next:
      self.next.emit()

    if (self.opcode == OP_BEGIN):
      retObj = (self.op_buf.getvalue(), self.weight_buf.getvalue(), self.data_buf.getvalue())
      self.op_buf.close()
      self.weight_buf.close()
      self.data_buf.close()
      return retObj

  def link_next(self, next):
    self.next = next
    return self.op_buf, self.sram_buf, self.flash_buf

class tensor:
  '''
  '''
  def __init__(self, tensor, buf):
    self.tensor = tensor
    self.buf = buf
    dim = tensor.shape
    if len(dim) == 0:
      self.x = 1
      self.y = 1
    elif len(dim) == 1:
      self.x = dim[0]
      self.y = 1
    elif len(dim) == 2:
      self.x = dim[0]
      self.y = dim[1]
    else:
      raise Exception("Unsupported dimention")

  def emit(self):
    # always use int8 for now
    dtype = 0

    # Only compress if is large enough
    # 0: dense, 1: huffman, 2: ANS, 3: custom
    codec = 1 if (self.x*self.y > 256) else 0

    # codec (2), dtype (2), bin (4)
    self.buf.write(((0<<6)|(0<<4)|(0)).to_bytes(1, 'little'))
    self.buf.write(self.x.to_bytes(2, 'little'))
    self.buf.write(self.y.to_bytes(2, 'little'))
    # Dense or compressed
    if (codec == 0):
      self.buf.write((self.x*self.y).to_bytes(3, 'little'))
      self.buf.write(self.tensor.numpy().tobytes())
    else: # (codec == 1): compressed
      zipper.compress_nparr(self.tensor.numpy(), 16)

