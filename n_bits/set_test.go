// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package n_bits

import (
	"strconv"
	"testing"
)

func TestBitSet(t *testing.T) {
	// Test Resize
	b := &BitSet{}
	for l := 60; l < 64*3+2; l++ {
		t.Run(strconv.Itoa(l), func(t *testing.T) {
			b.Resize(l)
			if b.Len != l {
				t.Errorf("expected length 100, got %d", b.Len)
			}

			// Test Set and Get
			b.Set(10)
			b.Set(50)
			if l > 99 {
				b.Set(99)
			}
			if !b.Get(10) {
				t.Errorf("expected bit 10 to be set")
			}
			if !b.Get(50) {
				t.Errorf("expected bit 50 to be set")
			}
			if l > 99 {
				if !b.Get(99) {
					t.Errorf("expected bit 99 to be set")
				}
			}
			if b.Get(0) {
				t.Errorf("expected bit 0 to be unset")
			}
			if b.Get(9) {
				t.Errorf("expected bit 9 to be unset")
			}
			if b.Get(11) {
				t.Errorf("expected bit 11 to be unset")
			}
			if l > 99 {
				if b.Get(98) {
					t.Errorf("expected bit 98 to be unset")
				}
			}

			// Test Effective
			if l == 100 && b.Effective() != 3 {
				t.Errorf("expected 3 effective bits, got %d", b.Effective())
			}

			// Test Expand
			bits := b.Expand()
			if len(bits) != b.Len {
				t.Errorf("expected %d bits in expanded slice, got %d", b.Len, len(bits))
			}
			for i := range bits {
				if i == 10 || i == 50 || (l > 99 && i == 99) {
					if !bits[i] {
						t.Errorf("expected bit %d to be set", i)
					}
				} else {
					if bits[i] {
						t.Errorf("expected bit %d to be unset", i)
					}
				}
			}

			// Test JSON Marshal/Unmarshal
			jsonData, err := b.MarshalJSON()
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			var got BitSet
			if err := got.UnmarshalJSON(jsonData); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if got.Len != b.Len {
				t.Errorf("expected length %d, got %d\nb:   %+v\ngot: %+v", b.Len, got.Len, b, &got)
			}
			for i := 0; i < b.Len; i++ {
				if b.Get(i) != got.Get(i) {
					t.Errorf("bit %d mismatch", i)
				}
			}
		})
	}
}
