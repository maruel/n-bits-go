// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package floatx

import (
	"math"
	"strconv"
	"testing"
)

func Test_BF16_Components(t *testing.T) {
	data := []struct {
		v        uint16
		f        float32
		sign     uint8
		exponent uint8
		mantissa uint8
	}{
		{0x3F80, 1.0, 0, 127, 0},
		{0xBF80, -1.0, 1, 127, 0},
		{0x4000, 2.0, 0, 128, 0},
		{0x3F00, 0.5, 0, 126, 0},
		{0xBF00, -0.5, 1, 126, 0},
		{0x0000, 0.0, 0, 0, 0},
		{0xC2F7, -123.5, 1, 133, 119},
		{0x7F80, float32(math.Inf(0)), 0, 255, 0},
		{0xFF80, float32(math.Inf(-1)), 1, 255, 0},
		{0x7FC0, float32(math.NaN()), 0, 255, 64},
	}
	for i, line := range data {
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			bf := BF16(line.v)
			sign, exponent, mantissa := bf.Components()
			if sign != line.sign || exponent != line.exponent || mantissa != line.mantissa {
				t.Fatalf("%d == %d && %d == %d || %d == %d", sign, line.sign, exponent, line.exponent, mantissa, line.mantissa)
			}
			if actual := bf.Float32(); actual != line.f {
				if !math.IsNaN(float64(actual)) && !math.IsNaN(float64(line.f)) {
					t.Fatalf("%g != %g", actual, line.f)
				}
			}
		})
	}
}
