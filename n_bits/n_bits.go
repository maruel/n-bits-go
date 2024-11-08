// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package n_bits

import (
	"fmt"
	"math"
	"reflect"
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
	Avg      float32           `json:"avg"`
	Min      float32           `json:"min"`
	Max      float32           `json:"max"`
	Sign     BitKind           `json:"s"`
	Exponent BitKind           `json:"exp"`
	Mantissa BitKindBool       `json:"man"`
}

// Len returns the number of bytes this tensor occupies.
func (a *AnalyzedTensor) Len() int64 {
	return a.NumEl * int64(a.DType.Size())
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

type BitKind struct {
	// Allocation is the number of bits allocated for this kind of value (sign, exponent, mantissa).
	Allocation int `json:"alloc"`
	// ValuesSeen is all the different values seen in the tensor. Is at least 1 and at most 1<<Allocation.
	ValuesSeen []int `json:"seen"`

	initialized  bool
	effective    int
	actuallyUsed float32
	wasted       int
}

func (b *BitKind) cache() {
	if !b.initialized {
		b.effective = effective(b.ValuesSeen)
		a := math.Log2(float64(b.effective))
		b.actuallyUsed = float32(a)
		b.wasted = b.Allocation - int(math.Ceil(a))
		b.initialized = true
	}
}

func (b *BitKind) NumberDifferentValuesSeen() int {
	b.cache()
	return b.effective
}

func (b *BitKind) BitsActuallyUsed() float32 {
	b.cache()
	return b.actuallyUsed
}

func (b *BitKind) BitsWasted() int {
	b.cache()
	return b.wasted
}

type BitKindBool struct {
	// Allocation is the number of bits allocated for this kind of value (sign, exponent, mantissa).
	Allocation int `json:"alloc"`
	// ValuesSeen is all the different values seen in the tensor. Is at least 1 and at most 1<<Allocation.
	ValuesSeen []bool `json:"seen"`

	initialized  bool
	effective    int
	actuallyUsed float32
	wasted       int
}

func (b *BitKindBool) cache() {
	if !b.initialized {
		b.effective = effectiveBits(b.ValuesSeen)
		a := math.Log2(float64(b.effective))
		b.actuallyUsed = float32(a)
		b.wasted = b.Allocation - int(math.Ceil(a))
		b.initialized = true
	}
}

func (b *BitKindBool) NumberDifferentValuesSeen() int {
	b.cache()
	return b.effective
}

func (b *BitKindBool) BitsActuallyUsed() float32 {
	b.cache()
	return b.actuallyUsed
}

func (b *BitKindBool) BitsWasted() int {
	b.cache()
	return b.wasted
}

//

// effective returns the number of non-zero items in a slice.
func effective(l []int) int {
	o := 0
	for _, v := range l {
		if v != 0 {
			o++
		}
	}
	return o
}

// effectiveBits returns the number of non-zero items in a slice.
func effectiveBits(l []bool) int {
	o := 0
	for _, v := range l {
		if v {
			o++
		}
	}
	return o
}

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
func calcF16HistogramAndStats(t safetensors.TensorView) ([]int, []int, []bool, float32, float32, float32) {
	signs := [1 << 1]int{}
	exponents := [1 << 5]int{}
	mantissas := [1 << 10]bool{}
	min := float32(math.MaxFloat32)
	max := float32(-math.MaxFloat32)
	total := float32(0.)

	// Remapping the slice gives a significant performance boost (10%).
	data := t.Data
	// #nosec G103
	hdr := *(*reflect.SliceHeader)(unsafe.Pointer(&data))
	word := int(safetensors.F16.Size())
	hdr.Len /= word
	hdr.Cap /= word
	// #nosec G103
	mapped := *(*[]floatx.F16)(unsafe.Pointer(&hdr))
	numEl := len(mapped)
	for _, bf := range mapped {
		sign, exponent, mantissa := bf.Components()
		signs[sign]++
		exponents[exponent]++
		mantissas[mantissa] = true
		// This gives a small performance improvement (2%) over bf.Float32().
		v := f16Lookup[bf]
		total += v
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	return signs[:], exponents[:], mantissas[:], total / float32(numEl), min, max
}

// calcBF16HistogramAndStats calculates the actual use of sign, exponent and
// mantissa bits plus floating point stats.
func calcBF16HistogramAndStats(t safetensors.TensorView) ([]int, []int, []bool, float32, float32, float32) {
	signs := [1 << 1]int{}
	exponents := [1 << 8]int{}
	mantissas := [1 << 7]bool{}
	min := float32(math.MaxFloat32)
	max := float32(-math.MaxFloat32)
	total := float32(0.)

	// Remapping the slice gives a significant performance boost (10%).
	data := t.Data
	// #nosec G103
	hdr := *(*reflect.SliceHeader)(unsafe.Pointer(&data))
	word := int(safetensors.BF16.Size())
	hdr.Len /= word
	hdr.Cap /= word
	// #nosec G103
	mapped := *(*[]floatx.BF16)(unsafe.Pointer(&hdr))
	numEl := len(mapped)
	for _, bf := range mapped {
		sign, exponent, mantissa := bf.Components()
		signs[sign]++
		exponents[exponent]++
		mantissas[mantissa] = true
		// This gives a small performance improvement (2%) over bf.Float32().
		v := bf16Lookup[bf]
		total += v
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	return signs[:], exponents[:], mantissas[:], total / float32(numEl), min, max
}

// calcF32HistogramAndStats calculates the actual use of sign, exponent and
// mantissa bits plus floating point stats.
func calcF32HistogramAndStats(t safetensors.TensorView) ([]int, []int, []bool, float32, float32, float32) {
	const (
		// https://en.wikipedia.org/wiki/Single-precision_floating-point_format
		f32SignOffset     = 31
		f32ExponentOffset = 23
		exponentMask      = (1 << (f32SignOffset - f32ExponentOffset)) - 1
		mantissaMask      = (1 << f32ExponentOffset) - 1
	)
	signs := [1 << 1]int{}
	exponents := [1 << (f32SignOffset - f32ExponentOffset)]int{}
	// TODO: It's too much when processing >1000 tensors, like
	// stabilityai/stable-fast-3d. Use a bool.
	mantissas := [1 << f32ExponentOffset]bool{}
	min := float32(math.MaxFloat32)
	max := float32(-math.MaxFloat32)
	total := float32(0.)

	// Remapping the slice gives a significant performance boost (10%).
	data := t.Data
	// #nosec G103
	hdr := *(*reflect.SliceHeader)(unsafe.Pointer(&data))
	word := int(safetensors.F32.Size())
	hdr.Len /= word
	hdr.Cap /= word
	// #nosec G103
	mapped := *(*[]float32)(unsafe.Pointer(&hdr))
	numEl := len(mapped)
	for _, v := range mapped {
		b := math.Float32bits(v)
		sign := b >> f32SignOffset
		exponent := (b >> f32ExponentOffset) & exponentMask
		mantissa := b & mantissaMask
		signs[sign]++
		exponents[exponent]++
		mantissas[mantissa] = true
		total += v
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	return signs[:], exponents[:], mantissas[:], total / float32(numEl), min, max
}

// AnalyzeTensor analyzes how well used the bits in a tensor are used.
func AnalyzeTensor(name string, t safetensors.TensorView) (AnalyzedTensor, error) {
	switch t.DType {
	case safetensors.F16:
		signs, exponents, mantissas, avg, min, max := calcF16HistogramAndStats(t)
		analyzed := AnalyzedTensor{
			Name:     name,
			DType:    t.DType,
			NumEl:    int64(len(t.Data)) / int64(t.DType.Size()),
			Avg:      avg,
			Min:      min,
			Max:      max,
			Sign:     BitKind{Allocation: 1, ValuesSeen: signs},
			Exponent: BitKind{Allocation: 5, ValuesSeen: exponents},
			Mantissa: BitKindBool{Allocation: 10, ValuesSeen: mantissas},
		}
		return analyzed, nil
	case safetensors.BF16:
		signs, exponents, mantissas, avg, min, max := calcBF16HistogramAndStats(t)
		analyzed := AnalyzedTensor{
			Name:     name,
			DType:    t.DType,
			NumEl:    int64(len(t.Data)) / int64(t.DType.Size()),
			Avg:      avg,
			Min:      min,
			Max:      max,
			Sign:     BitKind{Allocation: 1, ValuesSeen: signs},
			Exponent: BitKind{Allocation: 8, ValuesSeen: exponents},
			Mantissa: BitKindBool{Allocation: 7, ValuesSeen: mantissas},
		}
		return analyzed, nil
	case safetensors.F32:
		signs, exponents, mantissas, avg, min, max := calcF32HistogramAndStats(t)
		analyzed := AnalyzedTensor{
			Name:     name,
			DType:    t.DType,
			NumEl:    int64(len(t.Data)) / int64(t.DType.Size()),
			Avg:      avg,
			Min:      min,
			Max:      max,
			Sign:     BitKind{Allocation: 1, ValuesSeen: signs},
			Exponent: BitKind{Allocation: 8, ValuesSeen: exponents},
			Mantissa: BitKindBool{Allocation: 23, ValuesSeen: mantissas},
		}
		return analyzed, nil
	default:
		return AnalyzedTensor{}, fmt.Errorf("%s: TODO implement support for dtype %s", name, t.DType)
	}
}
