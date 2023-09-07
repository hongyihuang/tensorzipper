import numpy as np
from numpy import inf
from matplotlib import pyplot
import io

def cumul_inverse(cumul_counts):
  lut = np.zeros(shape=(cumul_counts[-1]), dtype = int)
  j = 0
  for i in range(cumul_counts[-1]):
    if i < cumul_counts[j]:
      lut[i] = j-1
    else:
      j += 1
      lut[i] = j-1

  return lut

def h(x):
  bits = np.log2(x)
  bits[bits == -inf] = 0
  return bits*-x

def C_rANS(input, state, symbol_counts, total_counts, cumul_counts):
  s_count = symbol_counts[input] # current symbol count/frequency
  next_state = (state//s_count)*total_counts + cumul_counts[input] + (state % s_count)
  return next_state

def D_rANS(state, symbol_counts, total_counts, cumul_counts, cumul_inv_f):
  slot = state % total_counts # compute the slot
  s = cumul_inv_f[slot] # decode the symbol
  prev_state = (state//total_counts)*symbol_counts[s] + slot - cumul_counts[s] # update the state
  return s, prev_state

def Streaming_rANS_encoder(s_input, symbol_counts):
  bits = 8
  total_counts = np.sum(symbol_counts)  # Represents M
  cumul_counts = np.insert(np.cumsum(symbol_counts),0,0) # the cumulative frequencies
  bitstream = io.BytesIO() # initialize stream
  state = total_counts # state initialized to M
  #bitmask = 2**bits-1
  max_range = symbol_counts * (2**bits)

  for s in reversed(s_input): # iterate over the input
    # Output bits to the stream to bring the state in the range for the next encoding
    while state >= max_range[s]:
      bitstream.write(state.astype(np.uint8).tobytes())
      state = state >> bits
    state = C_rANS(s, state, symbol_counts, total_counts, cumul_counts) # The rANS encoding step

  bitstream.write(state.item().to_bytes(3, 'little'))

  retObj = bytes(reversed(bitstream.getvalue()))
  bitstream.close()
  return retObj

def Streaming_rANS_decoder(bitstream, size, symbol_counts):
  data = np.zeros(shape=(size), dtype = np.uint8)
  total_counts = np.sum(symbol_counts)  # Represents M
  cumul_counts = np.insert(np.cumsum(symbol_counts),0,0) #the cumulative frequencies
  cumul_inv_f = cumul_inverse(cumul_counts)
  i = 0
  j = 3
  streamLen = len(bitstream)
  state = int.from_bytes(bitstream[0:3], 'big')

  while (j < streamLen) or (state != total_counts):
    # perform the rANS decoding
    s, state = D_rANS(state, symbol_counts, total_counts, cumul_counts, cumul_inv_f)

    # remap the state into the acceptable range
    while (j < streamLen) and (state < total_counts):
      stream = bitstream[j]
      j += 1
      state = (state << 8) + stream

    data[i] = s
    i += 1

  return data

def gen_sample(symbol_counts, size, seed = 42):
  # Generate uniform RNG, then bin it accordingly
  rng = np.random.default_rng(seed=seed)
  rints = rng.integers(low=0, high=sum(symbol_counts), size=size)
  cum_sum = np.insert(np.cumsum(symbol_counts),0,0)
  table = cumul_inverse(cum_sum)
  return list(map(lambda i: table[i], rints))

def compress_nparr(nparr, bins):
  '''
  Input: 
  nparr - array to be compressed symmetrically
  bins - the number of bins to quantize
  '''
  # 1. Generate byte bins & symbol count to pass into
  (hist, t) = np.histogram(nparr, bins = bins)
  pyplot.stairs(hist, t)
  print(hist, len(hist))

  # round the freq distribution with sum equal to 256
  hist = np.ceil(hist/np.sum(hist)*256).astype(np.int32)
  err = 256 - np.sum(hist)
  hist[np.argmax(hist)] += err

  # make the zeros one
  hist[hist == 0] = 1
  err = 256 - np.sum(hist)
  hist[np.argmax(hist)] += err

  print(hist, len(hist))

  assert(np.sum(hist) == 256)

  # 2. Transform the input into uint8
  new_nparr = nparr.flatten() - np.min(nparr)
  print(np.min(new_nparr), np.max(new_nparr))

  # 3. Compress
  # WARNING: DO NOT PASS A SYMBOL WITH ZERO FREQUENCY, IT WILL INF LOOP
  # TEMPORARY FIX: +1 TO THE 0 FREQ SYM AND SUBTRACT MAX FREQ SYM
  bitstream = Streaming_rANS_encoder(new_nparr.view(np.uint8), hist)
  return (hist, bitstream)

  '''
  # 1. Generate byte bins & symbol count to pass into
  (hist, _) = np.histogram(nparr, bins = bins)
  err = 256 - np.sum(hist)
  hist[np.argmax(hist)] += err
  assert(np.sum(hist) == 256)

  # 2. Transform the input into uint8
  nparr = nparr.view(np.uint8) - np.min(nparr)

  # 3. Compress
  Streaming_rANS_encoder(nparr, hist)

  return (hist, bitstream)
  '''

def exportCArray(bitstream):
  '''
  Useful function to export an bitstream into C array string
  Needed for embeedded compiler
  '''
  buf = io.StringIO()
  buf.write('{')
  buf.write(', '.join(map(hex, bitstream)))
  buf.write('}')
  retObj = buf.getvalue()
  buf.close()
  return retObj

def theoreticalLimit(dist):
  '''
  Returns: theoretical limit compression of each element in bits

  Input: an array of symbol count or a frequency distribution
  (latter is a fp normalized vector, former is int counts)

  Example:
  theoreticalLimit([2, 5, 1]) 
  theoreticalLimit([2/8, 5/8, 1/8]) 
  '''
  nparr = np.array(dist)
  return sum(h(nparr/np.sum(nparr)))

'''
# Testing Script, to be migrated

# Bin according to symbol_count distribution
symbol_counts = np.array([5, 1, 2])
size = 1024**2

rand_samp = gen_sample(symbol_counts, size)
counts, bins = np.histogram(rand_samp, bins = len(symbol_counts))

print(sum(symbol_counts), counts, bins)
original_bits = np.log(3)/np.log(2)
compress_bits = sum(h(np.array(symbol_counts)/8))
print("Per element original bits: ", original_bits)
print("Per element compressed bits: ", compress_bits)
print("Ideal Total Bytes: ", compress_bits*size/8)
print("Ideal Ratio: ", compress_bits/original_bits)

bitstream = Streaming_rANS_encoder(rand_samp, symbol_counts)
out_symbols = Streaming_rANS_decoder(bitstream, size, symbol_counts)
print(all(out_symbols == rand_samp))

original_bytes = np.log(3)/np.log(2)*size/8
print("Original: ", original_bytes, "bytes")
print("Compressed: ", len(bitstream), "bytes")
print("Compression Ratio: ", len(bitstream)/original_bytes)
'''