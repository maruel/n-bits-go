// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"

	"github.com/maruel/floatx"
	"github.com/maruel/safetensors"
	"github.com/maruel/sillybot/huggingface"
)

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

func humanBytes(i int) string {
	switch {
	case i > 1024*1024*1024:
		return fmt.Sprintf("%.1fGiB", float64(i)/1024./1024./1024.)
	case i > 1024*1024:
		return fmt.Sprintf("%.1fMiB", float64(i)/1024./1024.)
	case i > 1024:
		return fmt.Sprintf("%.1fkiB", float64(i)/1024.)
	default:
		return fmt.Sprintf("%dB", i)
	}
}

func analyze(ctx context.Context, hfToken, hfRepo string) error {
	// TODO: The library kinda sucks for downloading and caching.
	cache := filepath.Join(".cache", hfRepo)
	if err := os.MkdirAll(cache, 0o777); err != nil {
		return err
	}
	hf, err := huggingface.New(hfToken, cache)
	if err != nil {
		return err
	}
	if hfRepo != "" {
		p, err := hf.EnsureFile(ctx, huggingface.PackedFileRef("hf:"+hfRepo+"/HEAD/model.safetensors"), 0o666)
		if err != nil {
			return err
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		s, err := safetensors.Deserialize(b)
		if err != nil {
			return err
		}
		tensors := s.Tensors()
		maxNameLen := 0
		maxSizeLen := 0
		for _, tensor := range tensors {
			if l := len(tensor.Name); l > maxNameLen {
				maxNameLen = l
			}
			if l := len(strconv.Itoa(int(tensor.TensorView.DataLen() / 2))); l > maxSizeLen {
				maxSizeLen = l
			}
		}
		totalBytes := 0
		bytesWasted := 0
		for _, tensor := range tensors {
			t := tensor.TensorView
			name := tensor.Name
			if dt := t.DType(); dt != safetensors.BF16 {
				return fmt.Errorf("%s: can't handle dtype %s", name, dt)
			}
			numEl := int(t.DataLen() / 2)
			signs, exponents, mantissas := calcBF16Histogram(t)
			avg, min, max := calcBF16Stats(t)
			e_signs := effective(signs)
			e_exponents := effective(exponents)
			e_mantissas := effective(mantissas)
			b_signs := math.Log2(float64(e_signs))
			b_exponents := math.Log2(float64(e_exponents))
			b_mantissas := math.Log2(float64(e_mantissas))
			wasted := 1 - int(math.Ceil(b_signs)) + 8 - int(math.Ceil(b_exponents)) + 7 - int(math.Ceil(b_mantissas))
			fmt.Printf("%-*s: %*dw  avg=%4.1f [%6.1f, %6.1f]  sign=%1.0fbit  exponent=%3.1f/%dbits  mantissa=%3.1f/%dbits  wasted=%d/16bits %.1f%%  %8s\n",
				maxNameLen, name, maxSizeLen, numEl,
				avg, min, max,
				b_signs,
				b_exponents, 8,
				b_mantissas, 7,
				wasted, 100.*float64(wasted)/16., humanBytes(wasted*numEl/8),
			)
			/*
				fmt.Printf("%-*s: %*dw  sign=%d/%d %1.0fbits %3.0f%%  exponent=%3d/%d %3.1fbits %5.1f%%  mantissa=%3d/%d %3.1fbits %5.1f%%  wasted=%d/16bits %8dbytes %.1f%%\n",
					maxNameLen, name, maxSizeLen, numEl,
					e_signs, 1<<1, b_signs, float64(e_signs-1)/float64(1<<1-1)*100.,
					e_exponents, 1<<8, b_exponents, float64(e_exponents-1)/float64(1<<8-1)*100.,
					e_mantissas, 1<<7, b_mantissas, float64(e_mantissas-1)/float64(1<<7-1)*100.,
					wasted, wasted*numEl/8, 100.*float64(wasted)/16.,
				)
			*/
			bytesWasted += wasted * numEl / 8
			totalBytes += numEl * 2
		}
		fmt.Printf("%d bytes (%.1f%%) wasted on %d bytes total\n", bytesWasted, 100.*float64(bytesWasted)/float64(totalBytes), totalBytes)
	}
	return nil
}
