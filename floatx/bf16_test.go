// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package floatx

import (
	"fmt"
	"math"
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
		{0x3F00, 0.5, 0, 126, 0},
		{0xBF00, -0.5, 1, 126, 0},
		{0x4000, 2.0, 0, 128, 0},
		{0xC000, -2.0, 1, 128, 0},
		// https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Examples
		{0x0000, 0., 0, 0, 0},
		{0x8000, -0., 1, 0, 0},
		{0x7F7F, 3.3895314e+38, 0, 254, 127},
		{0x0080, 1.175494351e-38, 0, 1, 0},
		{0x4049, 3.140625, 0, 128, 73},    // pi
		{0x3EAB, 0.333984375, 0, 125, 43}, // 1/3
		{0x7F80, float32(math.Inf(0)), 0, 255, 0},
		{0xFF80, float32(math.Inf(-1)), 1, 255, 0},
		{0x7FC0, float32(math.NaN()), 0, 255, 64},
		{0xFFC1, float32(math.NaN()), 1, 255, 65}, // qNaN
		{0xFF81, float32(math.NaN()), 1, 255, 1},  // sNaN
	}
	for i, line := range data {
		t.Run(fmt.Sprintf("#%d: %g", i, line.f), func(t *testing.T) {
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
