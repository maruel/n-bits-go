// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package n_bits

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/maruel/floatx"
	"github.com/maruel/safetensors"
)

// AnalyzedModel is the analyzed data.
type AnalyzedModel struct {
	Tensors []AnalyzedTensor `json:"tensors"`
}

// AnalyzedTensor contains the stats coming from an analyzed tensor.
type AnalyzedTensor struct {
	Name     string            `json:"name"`
	DType    safetensors.DType `json:"dtype"`
	NumEl    int64             `json:"numel"` // Number of weights.
	Avg      float64           `json:"avg"`
	Min      float64           `json:"min"`
	Max      float64           `json:"max"`
	Inf      int               `json:"inf"`
	NaN      int               `json:"nan"`
	Sign     BitAllocation     `json:"s"`
	Exponent BitAllocation     `json:"exp"`
	Mantissa BitAllocation     `json:"man"`
}

// Len returns the number of bytes this tensor occupies.
func (a *AnalyzedTensor) Len() int64 {
	return a.NumEl * int64(a.DType.WordSize())
}

/* TODO
// IsFloat16Compatible returns true if the tensor can be represented as float16.
func (a *AnalyzedTensor) IsFloat16Compatible() bool {
	if a.DType != safetensors.BF16 {
		panic("implement me")
	}
	// Look if there's any exponent value that are outside of the range possible to float16.

	for i := 1; i < 10; i++ {
		if a.Exponent.ValuesSeen[i] != 0 {
			return false
		}
	}
	return true
}
*/

type BitAllocation interface {
	GetAllocation() int32
	NumberDifferentValuesSeen() int32
	BitsActuallyUsed() float64
	BitsWasted() int32
}

type BitKindCount struct {
	// Allocation is the number of bits allocated for this kind of value (sign, exponent, mantissa).
	Allocation int32 `json:"alloc"`
	// ValuesSeen is all the different values seen in the tensor. Is at least 1 and at most 1<<Allocation.
	ValuesSeen CountSet `json:"seen"`

	initialized  bool
	effective    int32
	actuallyUsed float64
	wasted       int32
}

func (b *BitKindCount) cache() {
	if !b.initialized {
		b.effective = b.ValuesSeen.Effective()
		a := math.Log2(float64(b.effective))
		b.actuallyUsed = a
		b.wasted = 0
		if b.Allocation != 0 {
			b.wasted = b.Allocation - int32(math.Ceil(a))
		}
		b.initialized = true
	}
}

func (b *BitKindCount) GetAllocation() int32 {
	return b.Allocation
}

func (b *BitKindCount) NumberDifferentValuesSeen() int32 {
	b.cache()
	return b.effective
}

func (b *BitKindCount) BitsActuallyUsed() float64 {
	b.cache()
	return b.actuallyUsed
}

func (b *BitKindCount) BitsWasted() int32 {
	b.cache()
	return b.wasted
}

type BitKindBool struct {
	// Allocation is the number of bits allocated for this kind of value (sign, exponent, mantissa).
	Allocation int32 `json:"alloc"`
	// ValuesSeen is all the different values seen in the tensor. Is at least 1 and at most 1<<Allocation.
	ValuesSeen BitSet `json:"seen"`

	initialized  bool
	effective    int32
	actuallyUsed float64
	wasted       int32
}

func (b *BitKindBool) cache() {
	if !b.initialized {
		b.effective = b.ValuesSeen.Effective()
		a := math.Log2(float64(b.effective))
		b.actuallyUsed = a
		b.wasted = 0
		if b.Allocation != 0 {
			b.wasted = b.Allocation - int32(math.Ceil(a))
		}
		b.initialized = true
	}
}

func (b *BitKindBool) GetAllocation() int32 {
	return b.Allocation
}

func (b *BitKindBool) NumberDifferentValuesSeen() int32 {
	b.cache()
	return b.effective
}

func (b *BitKindBool) BitsActuallyUsed() float64 {
	b.cache()
	return b.actuallyUsed
}

func (b *BitKindBool) BitsWasted() int32 {
	b.cache()
	return b.wasted
}

// BitMaskCount works for ints where the number of values is too large. Instead
// just look at the individual bits. It's not awesome but better than nothing.
type BitMaskCount struct {
	// Allocation is the number of bits allocated for this kind of value (sign, exponent, mantissa).
	Allocation int32 `json:"alloc"`
	// ValuesSeen is all the different values seen in the tensor. Is at least 1 and at most 1<<Allocation.
	ValuesSeen CountSet `json:"seen"`

	initialized  bool
	effective    int32
	actuallyUsed int32
	wasted       int32
}

func (b *BitMaskCount) cache() {
	if !b.initialized {
		// We don't log2() here.
		b.actuallyUsed = b.ValuesSeen.Effective()
		b.effective = 1 << b.actuallyUsed
		b.wasted = 0
		if b.Allocation != 0 {
			b.wasted = b.Allocation - b.actuallyUsed
		}
		b.initialized = true
	}
}

func (b *BitMaskCount) GetAllocation() int32 {
	return b.Allocation
}

func (b *BitMaskCount) NumberDifferentValuesSeen() int32 {
	b.cache()
	return b.effective
}

func (b *BitMaskCount) BitsActuallyUsed() float64 {
	b.cache()
	return float64(b.actuallyUsed)
}

func (b *BitMaskCount) BitsWasted() int32 {
	b.cache()
	return b.wasted
}

//

var f16Lookup [1 << 16]float32
var bf16Lookup [1 << 16]float32

func init() {
	for i := range bf16Lookup {
		f16Lookup[i] = floatx.F16(uint16(i)).Float32()
		bf16Lookup[i] = floatx.BF16(uint16(i)).Float32()
	}
}

// calcF16HistogramAndStats calculates the actual use of sign, exponent and
// mantissa bits plus floating point stats.
func calcF16HistogramAndStats(t safetensors.Tensor) (CountSet, CountSet, BitSet, float64, float64, float64, int, int) {
	var signs, exponents CountSet
	signs.Resize(1 << 1)
	exponents.Resize(1 << (floatx.F16SignOffset - floatx.F16ExponentOffset))
	var mantissas BitSet
	mantissas.Resize(1 << floatx.F16ExponentOffset)
	min := math.MaxFloat32
	max := -math.MaxFloat32
	total := 0.
	inf := 0
	nan := 0

	// Remapping the slice gives a significant performance boost (10%).
	// #nosec G103
	mapped := unsafe.Slice((*floatx.F16)(unsafe.Pointer(unsafe.SliceData(t.Data))), len(t.Data)/int(safetensors.F16.WordSize()))
	numEl := len(mapped)
	for _, bf := range mapped {
		sign, exponent, mantissa := bf.Components()
		signs.Add(int(sign))
		exponents.Add(int(exponent))
		mantissas.Set(int(mantissa))
		// The lookup gives a small performance improvement (2%) over f.Float32().
		// Consider anything in the 1e37 range infinity.
		if v := float64(f16Lookup[bf]); math.IsNaN(v) {
			nan++
		} else if math.IsInf(v, 0) || v < -1e37 && v > 1e37 {
			inf++
		} else {
			total += v
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
		}
	}
	return signs, exponents, mantissas, total / float64(numEl), min, max, inf, nan
}

// calcBF16HistogramAndStats calculates the actual use of sign, exponent and
// mantissa bits plus floating point stats.
func calcBF16HistogramAndStats(t safetensors.Tensor) (CountSet, CountSet, BitSet, float64, float64, float64, int, int) {
	var signs, exponents CountSet
	signs.Resize(1 << 1)
	exponents.Resize(1 << (floatx.BF16SignOffset - floatx.BF16ExponentOffset))
	var mantissas BitSet
	mantissas.Resize(1 << floatx.BF16ExponentOffset)
	min := math.MaxFloat32
	max := -math.MaxFloat32
	total := 0.
	inf := 0
	nan := 0

	// Remapping the slice gives a significant performance boost (10%).
	// #nosec G103
	mapped := unsafe.Slice((*floatx.BF16)(unsafe.Pointer(unsafe.SliceData(t.Data))), len(t.Data)/int(safetensors.BF16.WordSize()))
	numEl := len(mapped)
	for _, bf := range mapped {
		sign, exponent, mantissa := bf.Components()
		signs.Add(int(sign))
		exponents.Add(int(exponent))
		mantissas.Set(int(mantissa))
		// The lookup gives a small performance improvement (2%) over bf.Float32().
		// Consider anything in the 1e37 range infinity. This is necessary for Mistral-7B-v0.3.
		if v := float64(bf16Lookup[bf]); math.IsNaN(v) {
			nan++
		} else if math.IsInf(v, 0) || v < -1e37 || v > 1e37 {
			inf++
		} else {
			total += v
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
		}
	}
	return signs, exponents, mantissas, total / float64(numEl), min, max, inf, nan
}

// calcF32HistogramAndStats calculates the actual use of sign, exponent and
// mantissa bits plus floating point stats.
func calcF32HistogramAndStats(t safetensors.Tensor) (CountSet, CountSet, BitSet, float64, float64, float64, int, int) {
	var signs, exponents CountSet
	signs.Resize(1 << 1)
	exponents.Resize(1 << (floatx.F32SignOffset - floatx.F32ExponentOffset))
	var mantissas BitSet
	mantissas.Resize(1 << floatx.F32ExponentOffset)
	min := math.MaxFloat32
	max := -math.MaxFloat32
	total := 0.
	inf := 0
	nan := 0

	// Remapping the slice gives a significant performance boost (10%).
	// #nosec G103
	mapped := unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(t.Data))), len(t.Data)/int(safetensors.F32.WordSize()))
	numEl := len(mapped)
	for _, f := range mapped {
		b := math.Float32bits(f)
		sign := b >> floatx.F32SignOffset
		exponent := (b >> floatx.F32ExponentOffset) & floatx.F32ExponentMask
		mantissa := b & floatx.F32MantissaMask
		signs.Add(int(sign))
		exponents.Add(int(exponent))
		mantissas.Set(int(mantissa))
		// Consider anything in the 1e37 range infinity.
		if v := float64(f); math.IsNaN(v) {
			nan++
		} else if math.IsInf(v, 0) || v < -1e37 || v > 1e37 {
			inf++
		} else {
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
		}
	}
	return signs, exponents, mantissas, total / float64(numEl), min, max, inf, nan
}

// calcI32HistogramAndStats calculates the actual use of sign and mantissa bits
// plus stats.
//
// It does a very simplified analysis for now due to memory usage concern.
func calcI32HistogramAndStats(t safetensors.Tensor) (CountSet, CountSet, float64, int32, int32) {
	var min int32 = math.MaxInt32
	var max int32 = math.MinInt32
	var total int64
	signs := CountSet{}
	signs.Resize(1 << 1)
	mantissas := CountSet{}
	mantissas.Resize(31)
	// #nosec G103
	mapped := unsafe.Slice((*int32)(unsafe.Pointer(unsafe.SliceData(t.Data))), len(t.Data)/int(safetensors.I32.WordSize()))
	numEl := len(mapped)
	for _, i := range mapped {
		signs.Add(int(uint32(i) >> 31))
		for j := range 31 {
			if i&(1<<j) != 0 {
				mantissas.Add(j)
			}
		}
		total += int64(i)
		if i < min {
			min = i
		}
		if i > max {
			max = i
		}
	}
	avg := float64(total) / float64(numEl)
	return signs, mantissas, avg, min, max
}

// calcU32HistogramAndStats calculates the actual use of sign and mantissa bits
// plus stats.
//
// It does a very simplified analysis for now due to memory usage concern.
func calcU32HistogramAndStats(t safetensors.Tensor) (CountSet, float64, uint32, uint32) {
	var min uint32 = math.MaxUint32
	var max uint32 = 0
	var total uint64
	mantissas := CountSet{}
	mantissas.Resize(32)
	// #nosec G103
	mapped := unsafe.Slice((*uint32)(unsafe.Pointer(unsafe.SliceData(t.Data))), len(t.Data)/int(safetensors.U32.WordSize()))
	numEl := len(mapped)
	for _, i := range mapped {
		for j := range 32 {
			if i&(1<<j) != 0 {
				mantissas.Add(j)
			}
		}
		total += uint64(i)
		if i < min {
			min = i
		}
		if i > max {
			max = i
		}
	}
	avg := float64(total) / float64(numEl)
	return mantissas, avg, min, max
}

// AnalyzeTensor analyzes how well used the bits in a tensor are used.
func AnalyzeTensor(name string, t safetensors.Tensor) (AnalyzedTensor, error) {
	switch t.DType {
	case safetensors.F16:
		signs, exponents, mantissas, avg, min, max, inf, nan := calcF16HistogramAndStats(t)
		analyzed := AnalyzedTensor{
			Name:     name,
			DType:    t.DType,
			NumEl:    int64(len(t.Data)) / int64(t.DType.WordSize()),
			Avg:      avg,
			Min:      min,
			Max:      max,
			Inf:      inf,
			NaN:      nan,
			Sign:     &BitKindCount{Allocation: 1, ValuesSeen: signs},
			Exponent: &BitKindCount{Allocation: 5, ValuesSeen: exponents},
			Mantissa: &BitKindBool{Allocation: 10, ValuesSeen: mantissas},
		}
		return analyzed, nil
	case safetensors.BF16:
		signs, exponents, mantissas, avg, min, max, inf, nan := calcBF16HistogramAndStats(t)
		analyzed := AnalyzedTensor{
			Name:     name,
			DType:    t.DType,
			NumEl:    int64(len(t.Data)) / int64(t.DType.WordSize()),
			Avg:      avg,
			Min:      min,
			Max:      max,
			Inf:      inf,
			NaN:      nan,
			Sign:     &BitKindCount{Allocation: 1, ValuesSeen: signs},
			Exponent: &BitKindCount{Allocation: 8, ValuesSeen: exponents},
			Mantissa: &BitKindBool{Allocation: 7, ValuesSeen: mantissas},
		}
		return analyzed, nil
	case safetensors.F32:
		signs, exponents, mantissas, avg, min, max, inf, nan := calcF32HistogramAndStats(t)
		analyzed := AnalyzedTensor{
			Name:     name,
			DType:    t.DType,
			NumEl:    int64(len(t.Data)) / int64(t.DType.WordSize()),
			Avg:      avg,
			Min:      min,
			Max:      max,
			Inf:      inf,
			NaN:      nan,
			Sign:     &BitKindCount{Allocation: 1, ValuesSeen: signs},
			Exponent: &BitKindCount{Allocation: 8, ValuesSeen: exponents},
			Mantissa: &BitKindBool{Allocation: 23, ValuesSeen: mantissas},
		}
		return analyzed, nil
	case safetensors.I32:
		// Used in AWQ and GPTQ.
		signs, mantissas, avg, min, max := calcI32HistogramAndStats(t)
		analyzed := AnalyzedTensor{
			Name:     name,
			DType:    t.DType,
			NumEl:    int64(len(t.Data)) / int64(t.DType.WordSize()),
			Avg:      avg,
			Min:      float64(min),
			Max:      float64(max),
			Inf:      0,
			NaN:      0,
			Sign:     &BitKindCount{Allocation: 1, ValuesSeen: signs},
			Exponent: &BitKindCount{Allocation: 0},
			Mantissa: &BitMaskCount{Allocation: 31, ValuesSeen: mantissas},
		}
		return analyzed, nil
	case safetensors.U32:
		// Used in MLX.
		mantissas, avg, min, max := calcU32HistogramAndStats(t)
		analyzed := AnalyzedTensor{
			Name:     name,
			DType:    t.DType,
			NumEl:    int64(len(t.Data)) / int64(t.DType.WordSize()),
			Avg:      avg,
			Min:      float64(min),
			Max:      float64(max),
			Inf:      0,
			NaN:      0,
			Sign:     &BitKindCount{Allocation: 0},
			Exponent: &BitKindCount{Allocation: 0},
			Mantissa: &BitMaskCount{Allocation: 32, ValuesSeen: mantissas},
		}
		return analyzed, nil
	default:
		return AnalyzedTensor{}, fmt.Errorf("%s: TODO implement support for dtype %s", name, t.DType)
	}
}
