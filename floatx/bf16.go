// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package floatx

import (
	"encoding/binary"
	"math"
)

const (
	// https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
	bf16SignOffset     = 15
	bf16ExponentOffset = 7
	bf16ExponentBias   = 127

	// https://en.wikipedia.org/wiki/Single-precision_floating-point_format
	f32SignOffset     = 31
	f32ExponentOffset = 23
	f32ExponentBias   = 127
)

// BF16 represents a google brain 16 float.
type BF16 uint16

// DecodeBF16 decode a little endian value.
func DecodeBF16(b []byte) BF16 {
	return BF16(binary.LittleEndian.Uint16(b))
}

// Components returns the sign, exponent and mantissa bits separated.
func (b BF16) Components() (uint8, uint8, uint8) {
	sign := b >> bf16SignOffset
	exponent := (b >> bf16ExponentOffset) & ((1 << (bf16SignOffset - bf16ExponentOffset)) - 1)
	mantissa := b & ((1 << bf16ExponentOffset) - 1)
	return uint8(sign), uint8(exponent), uint8(mantissa)
}

// Float32 returns the float32 equivalent.
func (b BF16) Float32() float32 {
	sign8, exponent8, mantissa8 := b.Components()
	// Realign sign right away.
	sign := uint32(sign8) << f32SignOffset
	exponent := uint32(exponent8)
	// Realign mantissa right away.
	mantissa := uint32(mantissa8) << (f32ExponentOffset - bf16ExponentOffset)
	if exponent == 0x1F {
		if mantissa == 0 {
			// Infinity.
			return math.Float32frombits(sign | 0x7F800000)
		}
		// NaN.
		return math.Float32frombits(sign | 0x7FC00000)
	}
	if exponent == 0 {
		if mantissa == 0 {
			// Zero.
			return math.Float32frombits(sign)
		}
		// Normalize subnormal numbers.
		exponent++
		for mantissa&0x7F800000 == 0 {
			mantissa <<= 1
			exponent--
		}
		mantissa &= 0x007FFFFF
	}
	exponent += f32ExponentBias - bf16ExponentBias
	return math.Float32frombits(sign | (exponent << f32ExponentOffset) | mantissa)
}
