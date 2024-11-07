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
	Mantissa BitKind           `json:"man"`
}

// Bytes returns the number of bytes this tensor occupies.
func (a *AnalyzedTensor) Bytes() int64 {
	if a.DType != safetensors.BF16 {
		return -1
	}
	return a.NumEl * 2
}

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

var bf16Lookup [1 << 16]float32

func init() {
	for i := range bf16Lookup {
		bf16Lookup[i] = floatx.BF16(uint16(i)).Float32()
	}
}

// calcBF16HistogramAndStats calculates the actual use of sign, exponent and
// mantissa bits plus floating point stats.
func calcBF16HistogramAndStats(t safetensors.TensorView) ([]int, []int, []int, float32, float32, float32) {
	signs := [1 << 1]int{}
	exponents := [1 << 8]int{}
	mantissas := [1 << 7]int{}
	min := float32(math.MaxFloat32)
	max := float32(-math.MaxFloat32)
	total := float32(0.)

	// Remapping the slice gives a significant performance boost (10%).
	data := t.Data
	hdr := *(*reflect.SliceHeader)(unsafe.Pointer(&data))
	hdr.Len /= 2
	hdr.Cap /= 2
	mapped := *(*[]floatx.BF16)(unsafe.Pointer(&hdr))
	numEl := len(mapped)
	for _, bf := range mapped {
		sign, exponent, mantissa := bf.Components()
		signs[sign]++
		exponents[exponent]++
		mantissas[mantissa]++
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

// AnalyzeTensor analyzes how well used the bits in a tensor are used.
func AnalyzeTensor(name string, t safetensors.TensorView) (AnalyzedTensor, error) {
	if dt := t.DType; dt != safetensors.BF16 {
		return AnalyzedTensor{}, fmt.Errorf("%s: TODO implement support for dtype %s", name, dt)
	}
	signs, exponents, mantissas, avg, min, max := calcBF16HistogramAndStats(t)
	analyzed := AnalyzedTensor{
		Name:     name,
		DType:    safetensors.BF16,
		NumEl:    int64(len(t.Data) / 2),
		Avg:      avg,
		Min:      min,
		Max:      max,
		Sign:     BitKind{Allocation: 1, ValuesSeen: signs},
		Exponent: BitKind{Allocation: 8, ValuesSeen: exponents},
		Mantissa: BitKind{Allocation: 7, ValuesSeen: mantissas},
	}
	return analyzed, nil
}
