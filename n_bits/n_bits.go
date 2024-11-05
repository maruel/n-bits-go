// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package n_bits

import (
	"fmt"
	"math"

	"github.com/maruel/floatx"
	"github.com/maruel/safetensors"
)

// AnalyzedModel is the analyzed data.
type AnalyzedModel struct {
	Tensors []AnalyzedTensor `json:"tensors"`
}

type AnalyzedTensor struct {
	Name     string  `json:"name"`
	DType    string  `json:"dtype"`
	NumEl    int64   `json:"numel"`
	Avg      float32 `json:"avg"`
	Min      float32 `json:"min"`
	Max      float32 `json:"max"`
	Sign     BitKind `json:"s"`
	Exponent BitKind `json:"exp"`
	Mantissa BitKind `json:"man"`
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

// calcBF16Histogram calculates the actual use of sign, exponent and mantissa bits.
func calcBF16Histogram(t safetensors.TensorView) ([]int, []int, []int) {
	data := t.Data()
	signs := [1 << 1]int{}
	exponents := [1 << 8]int{}
	mantissas := [1 << 7]int{}
	for i := range t.DataLen() / 2 {
		bf := floatx.DecodeBF16(data[2*i:])
		sign, exponent, mantissa := bf.Components()
		signs[sign]++
		exponents[exponent]++
		mantissas[mantissa]++
	}
	return signs[:], exponents[:], mantissas[:]
}

// calcBF16Stats calculates the average, min and max
func calcBF16Stats(t safetensors.TensorView) (float32, float32, float32) {
	numEl := t.DataLen() / 2
	min := float32(math.MaxFloat32)
	max := float32(-math.MaxFloat32)
	total := float32(0.)
	data := t.Data()
	for i := range numEl {
		bf := floatx.DecodeBF16(data[2*i:])
		v := bf.Float32()
		total += v
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	return total / float32(numEl), min, max
}

// AnalyzeTensor analyzes how well used the bits in a tensor are used.
func AnalyzeTensor(name string, t safetensors.TensorView) (AnalyzedTensor, error) {
	if dt := t.DType(); dt != safetensors.BF16 {
		return AnalyzedTensor{}, fmt.Errorf("%s: TODO implement support for dtype %s", name, dt)
	}
	signs, exponents, mantissas := calcBF16Histogram(t)
	avg, min, max := calcBF16Stats(t)
	analyzed := AnalyzedTensor{
		Name:     name,
		DType:    "bfloat16",
		NumEl:    int64(t.DataLen() / 2),
		Avg:      avg,
		Min:      min,
		Max:      max,
		Sign:     BitKind{Allocation: 1, ValuesSeen: signs},
		Exponent: BitKind{Allocation: 8, ValuesSeen: exponents},
		Mantissa: BitKind{Allocation: 7, ValuesSeen: mantissas},
	}
	return analyzed, nil
}
